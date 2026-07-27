from dataclasses import FrozenInstanceError

import numpy as np
import polars as pl
import pytest

from tracksdata.nodes._mask import (
    Mask,
    _nd_sphere,
    mask_bbox_struct_fields,
    mask_crop,
    mask_dilate,
    mask_from_coordinates,
    mask_from_struct,
    mask_indices,
    mask_intersection,
    mask_iou,
    mask_move,
    mask_paint_buffer,
    mask_regionprops,
    mask_size,
    mask_struct_dtype,
    mask_subtract,
    mask_to_struct,
    mask_union,
    masks_from_column,
)


def test_mask_init() -> None:
    """Test Mask initialization."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(bbox=bbox, mask=mask_array)
    assert np.array_equal(mask.mask, mask_array)
    assert np.array_equal(mask.bbox, bbox)


def test_mask_validates_on_construction() -> None:
    """`Mask` should reject bboxes that disagree with the binary array."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)

    # bbox with the wrong number of dimensions
    with pytest.raises(ValueError, match="does not match bbox dimension"):
        Mask(bbox=np.array([0, 0, 0, 2, 2, 2]), mask=mask_array)

    # bbox of the right dimension but the wrong size
    with pytest.raises(ValueError, match="does not match bbox size"):
        Mask(bbox=np.array([0, 0, 3, 3]), mask=mask_array)

    # bbox is normalized to an int64 array
    mask = Mask(bbox=[0, 0, 2, 2], mask=mask_array)
    assert mask.bbox.dtype == np.int64
    np.testing.assert_array_equal(mask.bbox, [0, 0, 2, 2])


def test_mask_is_frozen() -> None:
    """Masks are immutable, so the geometry functions cannot alias their input."""
    mask = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.ones((2, 2), dtype=bool))

    with pytest.raises(FrozenInstanceError):
        mask.bbox = np.array([1, 1, 3, 3])


def test_mask_regionprops_spacing_aware() -> None:
    """Regionprops should account for spacing when provided."""
    mask_array = np.array([[False, True], [True, False]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])
    spacing = np.array([2.0, 3.0])  # y-spacing=2.0, x-spacing=3.0

    props = mask_regionprops(Mask(bbox=bbox, mask=mask_array), spacing=spacing)

    assert props.area == 12  # area (2) should be scaled by spacing (2*3=6)
    np.testing.assert_allclose(props.centroid, np.array([1.0, 1.5]))


def test_mask_regionprops_bbox_aware() -> None:
    """Regionprops should return absolute coordinates using the bbox offset."""
    mask_array = np.array([[False, True], [True, False]], dtype=bool)
    bbox = np.array([5, 10, 7, 12])

    props = mask_regionprops(Mask(bbox=bbox, mask=mask_array))

    assert props.area == 2
    np.testing.assert_allclose(props.centroid, np.array([5.5, 10.5]))

    coords = np.array(sorted(map(tuple, props.coords.tolist())))
    expected_coords = np.array(sorted([(5, 11), (6, 10)]))
    np.testing.assert_array_equal(coords, expected_coords)


def test_mask_regionprops_bbox_aware_3d() -> None:
    """Regionprops should handle 3D masks and preserve absolute coordinates."""
    mask_array = np.zeros((2, 2, 2), dtype=bool)
    mask_array[0, 0, 1] = True
    mask_array[1, 1, 0] = True
    bbox = np.array([3, 4, 5, 5, 6, 7])

    props = mask_regionprops(Mask(bbox=bbox, mask=mask_array))

    assert props.area == 2
    np.testing.assert_allclose(props.centroid, np.array([3.5, 4.5, 5.5]))

    coords = np.array(sorted(map(tuple, props.coords.tolist())))
    expected_coords = np.array(sorted([(3, 4, 6), (4, 5, 5)]))
    np.testing.assert_array_equal(coords, expected_coords)


def test_mask_regionprops_empty() -> None:
    """Regionprops should raise for empty masks."""
    mask_array = np.zeros((2, 2), dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    with pytest.raises(ValueError, match="single region"):
        _ = mask_regionprops(Mask(bbox=bbox, mask=mask_array))


def test_mask_regionprops_intensity_image() -> None:
    """Regionprops should handle intensity images."""
    mask_array = np.array([[False, True], [True, False]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])
    intensity_image = np.array([[1, 2], [3, 4]])

    props = mask_regionprops(Mask(bbox=bbox, mask=mask_array), intensity_image=intensity_image)
    assert props.intensity_max == 3  # 4 is outside the mask
    assert props.intensity_min == 2  # 0 is outside the mask


def test_mask_indices_no_offset() -> None:
    """Test mask_indices with no offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 2, 3, 4])  # min_y, min_x, max_y, max_x

    mask = Mask(bbox=bbox, mask=mask_array)
    indices = mask_indices(mask)

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

    mask = Mask(bbox=bbox, mask=mask_array)
    indices = mask_indices(mask, offset=5)

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

    mask = Mask(bbox=bbox, mask=mask_array)
    offset = np.array([3, 4])
    indices = mask_indices(mask, offset=offset)

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

    mask = Mask(bbox=bbox, mask=mask_array)
    indices = mask_indices(mask)

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
    """Test mask_paint_buffer."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(bbox=bbox, mask=mask_array)

    # Create a buffer to paint on
    buffer = np.zeros((4, 4), dtype=float)
    mask_paint_buffer(mask, buffer, value=5.0)

    # Check that the correct positions are painted
    expected_buffer = np.zeros((4, 4), dtype=float)
    expected_buffer[0, 0] = 5.0  # First True position
    expected_buffer[1, 1] = 5.0  # Second True position

    assert np.array_equal(buffer, expected_buffer)


def test_paint_buffer_with_offset() -> None:
    """Test mask_paint_buffer with offset."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(bbox=bbox, mask=mask_array)

    # Create a buffer to paint on
    buffer = np.zeros((6, 6), dtype=float)
    offset = np.array([2, 3])
    mask_paint_buffer(mask, buffer, value=7.0, offset=offset)

    # Check that the correct positions are painted with offset
    expected_buffer = np.zeros((6, 6), dtype=float)
    expected_buffer[2, 3] = 7.0  # First True position + offset
    expected_buffer[3, 4] = 7.0  # Second True position + offset

    assert np.array_equal(buffer, expected_buffer)


def test_paint_buffer_bbox_completely_outside_positive() -> None:
    """Bbox is entirely beyond the buffer boundary (positive overhang) — nothing should be painted."""
    mask_array = np.array([[True, True], [True, True]], dtype=bool)
    bbox = np.array([10, 10, 12, 12])  # entirely outside a (5, 5) buffer

    mask = Mask(bbox=bbox, mask=mask_array)
    buffer = np.zeros((5, 5), dtype=float)
    mask_paint_buffer(mask, buffer, value=1.0)

    assert np.all(buffer == 0.0)


def test_paint_buffer_bbox_completely_outside_negative() -> None:
    """Bbox is entirely at negative coordinates — nothing should be painted (no wrap-around)."""
    mask_array = np.array([[True, True], [True, True]], dtype=bool)
    bbox = np.array([-5, -5, -3, -3])  # entirely negative

    mask = Mask(bbox=bbox, mask=mask_array)
    buffer = np.zeros((5, 5), dtype=float)
    mask_paint_buffer(mask, buffer, value=1.0)

    assert np.all(buffer == 0.0)


def test_paint_buffer_bbox_touching_positive_edge() -> None:
    """Bbox ends exactly at the buffer boundary — all pixels should be painted."""
    # 2x2 mask at [3, 3] -> [5, 5], fits exactly inside a (5, 5) buffer
    mask_array = np.array([[True, True], [True, True]], dtype=bool)
    bbox = np.array([3, 3, 5, 5])  # fits exactly in (5,5)

    mask = Mask(bbox=bbox, mask=mask_array)
    buffer = np.zeros((5, 5), dtype=float)
    mask_paint_buffer(mask, buffer, value=2.0)

    expected = np.zeros((5, 5), dtype=float)
    expected[3:5, 3:5] = 2.0
    assert np.array_equal(buffer, expected)


def test_paint_buffer_bbox_on_positive_edge_overhang() -> None:
    """Bbox starts inside the buffer but extends one row/col beyond the edge."""
    # 3x3 mask starting at (3,3), buffer is (5,5) → rows 5 and col 5 overhang by 1
    mask_array = np.ones((3, 3), dtype=bool)
    bbox = np.array([3, 3, 6, 6])

    mask = Mask(bbox=bbox, mask=mask_array)
    buffer = np.zeros((5, 5), dtype=float)
    mask_paint_buffer(mask, buffer, value=3.0)

    expected = np.zeros((5, 5), dtype=float)
    expected[3:5, 3:5] = 3.0  # only the 2x2 in-bounds portion
    assert np.array_equal(buffer, expected)


def test_paint_buffer_bbox_on_negative_edge_overhang() -> None:
    """Bbox starts before index 0 but ends inside the buffer — only in-bounds part painted."""
    # 3x3 mask with bbox [-1, -1, 2, 2]; only [0:2, 0:2] is in-bounds
    mask_array = np.ones((3, 3), dtype=bool)
    bbox = np.array([-1, -1, 2, 2])

    mask = Mask(bbox=bbox, mask=mask_array)
    buffer = np.zeros((5, 5), dtype=float)
    mask_paint_buffer(mask, buffer, value=4.0)

    expected = np.zeros((5, 5), dtype=float)
    expected[0:2, 0:2] = 4.0  # only the 2x2 in-bounds portion
    assert np.array_equal(buffer, expected)


def test_paint_buffer_bbox_on_negative_edge_with_sparse_mask() -> None:
    """Partial negative overhang with a sparse mask — only in-bounds True pixels painted."""
    # mask has True only at [0,0] and [2,2]; bbox is [-1,-1,2,2]
    # after clipping: [0,0] maps to buffer [-1+0,-1+0] = out of bounds → skipped
    #                 [2,2] maps to buffer [-1+2,-1+2] = [1,1] → painted
    mask_array = np.array([[True, False, False], [False, False, False], [False, False, True]], dtype=bool)
    bbox = np.array([-1, -1, 2, 2])

    mask = Mask(bbox=bbox, mask=mask_array)
    buffer = np.zeros((5, 5), dtype=float)
    mask_paint_buffer(mask, buffer, value=5.0)

    expected = np.zeros((5, 5), dtype=float)
    expected[1, 1] = 5.0
    assert np.array_equal(buffer, expected)


def test_mask_iou() -> None:
    """Test IoU calculation between masks."""
    # Create two overlapping masks
    mask1_array = np.array([[True, True], [True, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])
    mask1 = Mask(bbox=bbox1, mask=mask1_array)

    mask2_array = np.array([[True, False], [True, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])
    mask2 = Mask(bbox=bbox2, mask=mask2_array)

    iou = mask_iou(mask1, mask2)

    # Intersection: positions (0,0) and (1,0) = 2 pixels
    # Union: 3 + 3 - 2 = 4 pixels
    # IoU = 2/4 = 0.5
    expected_iou = 0.5
    assert abs(iou - expected_iou) < 1e-6


def test_mask_iou_no_overlap() -> None:
    """Test IoU calculation with non-overlapping masks."""
    mask1_array = np.array([[True, False], [False, False]], dtype=bool)
    bbox1 = np.array([0, 0, 2, 2])
    mask1 = Mask(bbox=bbox1, mask=mask1_array)

    mask2_array = np.array([[False, False], [False, True]], dtype=bool)
    bbox2 = np.array([0, 0, 2, 2])
    mask2 = Mask(bbox=bbox2, mask=mask2_array)

    iou = mask_iou(mask1, mask2)
    assert iou == 0.0


def test_mask_iou_identical() -> None:
    """Test IoU calculation with identical masks."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask1 = Mask(bbox=bbox, mask=mask_array)
    mask2 = Mask(bbox=bbox.copy(), mask=mask_array.copy())

    iou = mask_iou(mask1, mask2)
    assert iou == 1.0


def test_mask_intersection_and_size() -> None:
    """Intersection counts shared pixels; size counts the mask's own pixels."""
    mask1 = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.array([[True, True], [True, False]], dtype=bool))
    mask2 = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.array([[True, False], [True, True]], dtype=bool))

    assert mask_size(mask1) == 3
    assert mask_size(mask2) == 3
    assert mask_intersection(mask1, mask2) == 2


def test_mask_union_dimension_mismatch() -> None:
    """Mask union should raise when masks do not share the same dimensionality."""
    mask_2d = Mask(bbox=np.array([0, 0, 1, 2]), mask=np.ones((1, 2), dtype=bool))
    mask_3d = Mask(bbox=np.array([0, 0, 0, 1, 1, 2]), mask=np.ones((1, 1, 2), dtype=bool))

    with pytest.raises(ValueError, match=r"Cannot compute union between masks of different dimensions: 2 and 3."):
        _ = mask_union(mask_2d, mask_3d)


def test_mask_union_overlapping() -> None:
    """Mask union should merge overlapping masks into a single bounding box."""
    mask1_array = np.array([[True, True], [False, False]], dtype=bool)
    mask2_array = np.array([[False, True], [True, True]], dtype=bool)

    mask1 = Mask(bbox=np.array([0, 0, 2, 2]), mask=mask1_array)
    mask2 = Mask(bbox=np.array([1, 1, 3, 3]), mask=mask2_array)

    union = mask_union(mask1, mask2)

    expected_bbox = np.array([0, 0, 3, 3])
    expected_mask = np.array(
        [
            [True, True, False],
            [False, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )

    assert np.array_equal(union.bbox, expected_bbox)
    assert np.array_equal(union.mask, expected_mask)


def test_mask_union_disjoint() -> None:
    """Mask union should include both masks even when disjoint."""
    mask1_array = np.array([[True, True], [True, True]], dtype=bool)
    mask2_array = np.array([[True, False], [False, False]], dtype=bool)

    mask1 = Mask(bbox=np.array([0, 0, 2, 2]), mask=mask1_array)
    mask2 = Mask(bbox=np.array([3, 3, 5, 5]), mask=mask2_array)

    union = mask_union(mask1, mask2)
    reverse_union = mask_union(mask2, mask1)

    expected_bbox = np.array([0, 0, 5, 5])
    expected_mask = np.zeros((5, 5), dtype=bool)
    expected_mask[:2, :2] = True
    expected_mask[3, 3] = True

    assert np.array_equal(union.bbox, expected_bbox)
    assert np.array_equal(union.mask, expected_mask)
    assert union == reverse_union


def test_mask_equal() -> None:
    """`==` compares both the bbox and the binary array."""
    mask = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.ones((2, 2), dtype=bool))

    assert mask == (Mask(bbox=np.array([0, 0, 2, 2]), mask=np.ones((2, 2), dtype=bool)))
    assert mask != (Mask(bbox=np.array([1, 1, 3, 3]), mask=np.ones((2, 2), dtype=bool)))
    assert mask != (Mask(bbox=np.array([0, 0, 2, 2]), mask=np.zeros((2, 2), dtype=bool)))


def test_mask_empty() -> None:
    """Test mask with no True values."""
    mask_array = np.array([[False, False], [False, False]], dtype=bool)
    bbox = np.array([0, 0, 2, 2])

    mask = Mask(bbox=bbox, mask=mask_array)
    indices = mask_indices(mask)

    # Should return empty arrays
    assert len(indices) == 2
    assert len(indices[0]) == 0
    assert len(indices[1]) == 0


def test_mask_all_true() -> None:
    """Test mask with all True values."""
    mask_array = np.array([[True, True], [True, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    mask = Mask(bbox=bbox, mask=mask_array)
    indices = mask_indices(mask)

    # Should return all positions
    expected_y = np.array([1, 1, 2, 2])
    expected_x = np.array([1, 2, 1, 2])

    assert len(indices) == 2
    assert np.array_equal(indices[0], expected_y)
    assert np.array_equal(indices[1], expected_x)


def test_mask_crop() -> None:
    """Test mask cropping."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    mask = Mask(bbox=bbox, mask=mask_array)
    image = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    cropped_image = mask_crop(mask, image)
    assert np.array_equal(cropped_image, image[1:3, 1:3])


def test_mask_crop_with_shape() -> None:
    """Test mask cropping with shape."""
    mask_array = np.array([[True, False], [False, True]], dtype=bool)
    bbox = np.array([1, 1, 3, 3])

    mask = Mask(bbox=bbox, mask=mask_array)
    image = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    cropped_image = mask_crop(mask, image, shape=(2, 4))
    assert np.array_equal(cropped_image, image[1:3, 0:4])
    assert cropped_image.shape == (2, 4)

    cropped_image = mask_crop(mask, image, shape=(1, 4))
    assert np.array_equal(cropped_image, image[2:3, 0:4])
    assert cropped_image.shape == (1, 4)


def test_mask_from_coordinates_2d_basic() -> None:
    """Test 2D mask creation and bbox without cropping."""
    center = np.asarray([5, 5])
    radius = 2
    mask = mask_from_coordinates(center, radius)
    # Should be a disk of radius 2, shape (5,5), centered at (5,5)
    assert mask.mask.shape == (5, 5)
    assert mask.mask[2, 2]  # center pixel is True
    assert mask.mask.dtype == bool
    np.testing.assert_array_equal(mask.bbox, [3, 3, 8, 8])


def test_mask_from_coordinates_3d_basic() -> None:
    """Test 3D mask creation and bbox without cropping."""
    center = np.asarray([4, 5, 6])
    radius = 1
    mask = mask_from_coordinates(center, radius)
    # Should be a ball of radius 1, shape (3,3,3), centered at (4,5,6)
    assert mask.mask.shape == (3, 3, 3)
    assert mask.mask[1, 1, 1]  # center voxel is True
    np.testing.assert_array_equal(mask.bbox, [3, 4, 5, 6, 7, 8])


def test_mask_from_coordinates_cropping() -> None:
    """Test cropping when mask falls outside the image boundary."""
    center = np.asarray([0, 0])
    radius = 5
    image_shape = (4, 3)

    mask = mask_from_coordinates(center, radius, image_shape=image_shape)

    # Mask shape should match the bbox size
    expected_shape = (4, 3)
    assert mask.mask.shape == expected_shape

    # Mask should be cropped to fit within image bounds
    np.testing.assert_array_equal(mask.bbox, [0, 0, 4, 3])


def test_mask_simple_difference() -> None:
    """Test mask difference."""
    mask1_array = np.asarray([[[True, True], [True, False]]], dtype=bool)
    mask2_array = np.asarray([[[True, False], [True, True]]], dtype=bool)

    mask1 = Mask(bbox=np.asarray([0, 0, 0, 1, 2, 2]), mask=mask1_array)
    mask2 = Mask(bbox=np.asarray([0, 0, 0, 1, 2, 2]), mask=mask2_array)

    diff = mask_subtract(mask1, mask2)
    np.testing.assert_array_equal(diff.mask, np.asarray([[[False, True], [False, False]]], dtype=bool))
    np.testing.assert_array_equal(diff.bbox, np.asarray([0, 0, 0, 1, 2, 2]))

    # the input masks are left untouched
    np.testing.assert_array_equal(mask1.mask, mask1_array)


def test_mask_difference_no_overlap() -> None:
    """Test mask difference with no overlap."""
    mask1_array = np.asarray([[[True, True], [True, False]]], dtype=bool)
    mask2_array = np.asarray([[[False, False], [False, True]]], dtype=bool)

    mask1 = Mask(bbox=np.asarray([0, 0, 0, 1, 2, 2]), mask=mask1_array)
    mask2 = Mask(bbox=np.asarray([0, 0, 0, 1, 2, 2]), mask=mask2_array)

    diff = mask_subtract(mask1, mask2)
    np.testing.assert_array_equal(diff.mask, mask1_array)
    np.testing.assert_array_equal(diff.bbox, np.asarray([0, 0, 0, 1, 2, 2]))


def test_mask_difference_complex_overlap() -> None:
    mask1_array = np.asarray([[[True, True], [True, True]]], dtype=bool)
    mask2_array = np.asarray([[[True, True], [True, True]]], dtype=bool)

    mask1 = Mask(bbox=np.asarray([1, 3, 3, 2, 5, 5]), mask=mask1_array.copy())
    mask2 = Mask(bbox=np.asarray([1, 4, 4, 2, 6, 6]), mask=mask2_array)

    diff = mask_subtract(mask1, mask2)
    np.testing.assert_array_equal(diff.mask, np.asarray([[[True, True], [True, False]]], dtype=bool))
    np.testing.assert_array_equal(diff.bbox, np.asarray([1, 3, 3, 2, 5, 5]))

    # identical to mask1 checking reverse overlap
    mask3 = Mask(bbox=np.asarray([1, 3, 3, 2, 5, 5]), mask=mask1_array)
    reverse_diff = mask_subtract(mask2, mask3)

    np.testing.assert_array_equal(reverse_diff.mask, np.asarray([[[False, True], [True, True]]], dtype=bool))
    np.testing.assert_array_equal(reverse_diff.bbox, np.asarray([1, 4, 4, 2, 6, 6]))


def test_dilation_simple() -> None:
    point = np.asarray([[True]])
    bbox = np.asarray([5, 5, 6, 6])
    mask = Mask(bbox=bbox, mask=point)

    dilated = mask_dilate(mask, radius=2)

    np.testing.assert_array_equal(dilated.bbox, [3, 3, 8, 8])
    np.testing.assert_array_equal(
        dilated.mask,
        _nd_sphere(2, 2),
    )

    # the input mask is left untouched
    np.testing.assert_array_equal(mask.bbox, [5, 5, 6, 6])


def test_dilation_on_border() -> None:
    point = np.asarray([[True]])
    bbox = np.asarray([0, 0, 1, 1])

    # left overhang
    mask = Mask(bbox=bbox, mask=point)
    dilated = mask_dilate(mask, radius=2, image_shape=(7, 7))

    np.testing.assert_array_equal(dilated.bbox, [0, 0, 3, 3])
    np.testing.assert_array_equal(
        dilated.mask,
        _nd_sphere(2, 2)[2:, 2:],
    )

    bbox = np.asarray([6, 6, 7, 7])
    mask = Mask(bbox=bbox, mask=point)

    # right overhang
    dilated = mask_dilate(mask, radius=2, image_shape=(7, 7))
    np.testing.assert_array_equal(dilated.bbox, [4, 4, 7, 7])
    np.testing.assert_array_equal(
        dilated.mask,
        _nd_sphere(2, 2)[:-2, :-2],
    )


def test_mask_move() -> None:
    point = np.asarray([[True]])
    bbox = np.asarray([0, 0, 1, 1])
    mask = Mask(bbox=bbox, mask=point)

    moved = mask_move(mask, offset=np.asarray([5, 2]), image_shape=(7, 7))
    np.testing.assert_array_equal(moved.bbox, [5, 2, 6, 3])
    np.testing.assert_array_equal(moved.mask, point)

    # the input mask is left untouched
    np.testing.assert_array_equal(mask.bbox, [0, 0, 1, 1])

    moved = mask_move(moved, offset=np.asarray([-3, 2]), image_shape=(7, 7))
    np.testing.assert_array_equal(moved.bbox, [2, 4, 3, 5])
    np.testing.assert_array_equal(moved.mask, point)


def test_mask_struct_dtype() -> None:
    dtype_2d = mask_struct_dtype(2)
    assert dtype_2d == pl.Struct(
        {
            "min_y": pl.Int64,
            "min_x": pl.Int64,
            "max_y": pl.Int64,
            "max_x": pl.Int64,
            "data": pl.Binary,
        }
    )

    dtype_3d = mask_struct_dtype(3)
    assert [f.name for f in dtype_3d.fields] == [
        "min_z",
        "min_y",
        "min_x",
        "max_z",
        "max_y",
        "max_x",
        "data",
    ]

    assert mask_bbox_struct_fields(2) == ["min_y", "min_x", "max_y", "max_x"]

    with pytest.raises(ValueError):
        mask_struct_dtype(4)


@pytest.mark.parametrize("ndim", [2, 3])
def test_mask_struct_roundtrip(ndim: int) -> None:
    rng = np.random.default_rng(0)
    shape = (3, 4, 5)[:ndim]
    mask_data = rng.uniform(size=shape) > 0.5
    bbox = np.concatenate([np.arange(1, ndim + 1), np.arange(1, ndim + 1) + shape])

    mask = Mask(bbox=bbox, mask=mask_data)
    value = mask_to_struct(mask)

    assert isinstance(value, dict)
    assert isinstance(value["data"], bytes)
    assert value["min_y" if ndim == 2 else "min_z"] == 1

    restored = mask_from_struct(value)
    assert restored == mask


def test_masks_from_column() -> None:
    """`masks_from_column` decodes both struct and object mask columns."""
    mask = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.ones((2, 2), dtype=bool))

    struct_column = pl.Series([mask_to_struct(mask)], dtype=mask_struct_dtype(2))
    (from_struct_column,) = masks_from_column(struct_column)
    assert from_struct_column == mask

    object_column = pl.Series([mask], dtype=pl.Object)
    (from_object_column,) = masks_from_column(object_column)
    assert from_object_column is mask

    with pytest.raises(TypeError):
        masks_from_column(pl.Series(["not a mask"]))


def test_mask_struct_attr_in_graph(graph_backend) -> None:
    """Masks stored as struct attributes round-trip and are filterable by bbox fields."""

    from tracksdata.attrs import NodeAttr
    from tracksdata.constants import DEFAULT_ATTR_KEYS

    graph = graph_backend
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, mask_struct_dtype(2))

    mask_a = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.ones((2, 2), dtype=bool))
    mask_b = Mask(bbox=np.array([5, 6, 7, 9]), mask=np.ones((2, 3), dtype=bool))

    node_a = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask_to_struct(mask_a)})
    node_b = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, DEFAULT_ATTR_KEYS.MASK: mask_to_struct(mask_b)})

    df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK])
    assert df[DEFAULT_ATTR_KEYS.MASK].dtype == mask_struct_dtype(2)

    restored = dict(zip(df[DEFAULT_ATTR_KEYS.NODE_ID], masks_from_column(df[DEFAULT_ATTR_KEYS.MASK]), strict=True))
    assert restored[node_a] == mask_a
    assert restored[node_b] == mask_b

    # filtering on a bbox field of the mask struct
    filtered = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.MASK).struct.field("min_y") > 2).node_ids()
    assert filtered == [node_b]
