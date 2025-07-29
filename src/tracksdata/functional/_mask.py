"""
Utility functions for working with mask and bounding box arrays.
"""

import numpy as np
from numpy.typing import NDArray

from tracksdata.functional._iou import fast_intersection_with_bbox, fast_iou_with_bbox


def crop_image_with_bbox(
    image: NDArray,
    bbox: NDArray[np.integer],
    shape: tuple[int, ...] | None = None,
) -> NDArray:
    """
    Crop an image using bounding box coordinates.

    Parameters
    ----------
    image : NDArray
        The image to crop from.
    bbox : NDArray[np.integer]
        The bounding box coordinates with shape (2 * ndim,).
        First ndim elements are start indices, last ndim elements are end indices.
    shape : tuple[int, ...] | None, optional
        The shape of the cropped image. If None, the bbox will be used.

    Returns
    -------
    NDArray
        The cropped image.

    Raises
    ------
    ValueError
        If bbox length is not even or image dimensions don't match expected bbox dimensions.

    Examples
    --------
    Crop a 2D image using a bounding box:

    ```python
    import numpy as np

    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    bbox = np.array([1, 1, 3, 3])  # [min_y, min_x, max_y, max_x]
    cropped = crop_image_with_bbox(image, bbox)
    cropped
    array([[6, 7], [10, 11]])
    ```

    Crop with a fixed output shape:
    ```python
    cropped_fixed = crop_image_with_bbox(image, bbox, shape=(1, 4))
    cropped_fixed.shape
    (1, 4)
    ```
    """
    if len(bbox) % 2 != 0:
        raise ValueError(f"Bbox must have even length, got {len(bbox)}")

    ndim = len(bbox) // 2

    if image.ndim != ndim:
        raise ValueError(f"Image dimensions ({image.ndim}) must match bbox dimensions ({ndim})")

    # Validate bbox coordinates
    start_coords = bbox[:ndim]
    end_coords = bbox[ndim:]

    if np.any(start_coords < 0) or np.any(end_coords <= start_coords):
        raise ValueError(f"Invalid bbox coordinates: {bbox}")

    if shape is None:
        slicing = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))
    else:
        if len(shape) != ndim:
            raise ValueError(f"Shape dimensions ({len(shape)}) must match bbox dimensions ({ndim})")

        center = (bbox[:ndim] + bbox[ndim:]) // 2
        half_shape = np.asarray(shape) // 2
        start = np.maximum(center - half_shape, 0)
        end = np.minimum(center + half_shape, image.shape)
        slicing = tuple(slice(s, e) for s, e in zip(start, end, strict=True))

    return image[slicing]


def mask_indices(
    bbox: NDArray[np.integer],
    mask: NDArray[np.bool_],
    offset: NDArray[np.integer] | int = 0,
) -> tuple[NDArray[np.integer], ...]:
    """
    Get the indices of pixels that are part of the mask in global coordinates.

    Parameters
    ----------
    bbox : NDArray[np.integer]
        The bounding box coordinates with shape (2 * ndim,).
    mask : NDArray[np.bool_]
        Binary mask indicating valid pixels.
    offset : NDArray[np.integer] | int, optional
        Additional offset to add to the indices.

    Returns
    -------
    tuple[NDArray[np.integer], ...]
        The indices of pixels that are part of the mask.

    Raises
    ------
    ValueError
        If bbox length is not even or mask dimensions don't match expected bbox dimensions.

    Examples
    --------
    Get global indices for a 2D mask:

    >>> import numpy as np
    >>> mask = np.array([[True, False], [False, True]])
    >>> bbox = np.array([10, 20, 12, 22])  # [min_y, min_x, max_y, max_x]
    >>> y_indices, x_indices = mask_indices(bbox, mask)
    >>> y_indices
    array([10, 11])
    >>> x_indices
    array([20, 21])

    With an offset:

    >>> y_indices, x_indices = mask_indices(bbox, mask, offset=5)
    >>> y_indices
    array([15, 16])
    >>> x_indices
    array([25, 26])
    """
    if len(bbox) % 2 != 0:
        raise ValueError(f"Bbox must have even length, got {len(bbox)}")

    ndim = len(bbox) // 2

    if mask.ndim != ndim:
        raise ValueError(f"Mask dimensions ({mask.ndim}) must match bbox dimensions ({ndim})")

    if isinstance(offset, int):
        offset = np.full(ndim, offset)

    indices = list(np.nonzero(mask))

    for i, index in enumerate(indices):
        indices[i] = index + bbox[i] + offset[i]

    return tuple(indices)


def paint_mask_to_buffer(
    buffer: np.ndarray,
    bbox: NDArray[np.integer],
    mask: NDArray[np.bool_],
    value: int | float,
    offset: NDArray[np.integer] | int = 0,
) -> None:
    """
    Paint mask pixels into a buffer.

    Parameters
    ----------
    buffer : np.ndarray
        The buffer to paint into (modified in place).
    bbox : NDArray[np.integer]
        The bounding box coordinates.
    mask : NDArray[np.bool_]
        Binary mask indicating pixels to paint.
    value : int | float
        The value to paint.
    offset : NDArray[np.integer] | int, optional
        Additional offset to add to the indices.

    Raises
    ------
    ValueError
        If buffer dimensions don't match bbox dimensions or if bbox is invalid.

    Examples
    --------
    Paint mask pixels into a buffer:

    >>> import numpy as np
    >>> buffer = np.zeros((5, 5))
    >>> mask = np.array([[True, False], [False, True]])
    >>> bbox = np.array([1, 1, 3, 3])  # [min_y, min_x, max_y, max_x]
    >>> paint_mask_to_buffer(buffer, bbox, mask, value=255)
    >>> buffer[1:3, 1:3]
    array([[255.,   0.],
           [  0., 255.]])
    """
    if len(bbox) % 2 != 0:
        raise ValueError(f"Bbox must have even length, got {len(bbox)}")

    ndim = len(bbox) // 2

    if buffer.ndim != ndim:
        raise ValueError(f"Buffer dimensions ({buffer.ndim}) must match bbox dimensions ({ndim})")
    indices = mask_indices(bbox, mask, offset)
    buffer[indices] = value


def mask_iou(
    bbox1: NDArray[np.integer],
    mask1: NDArray[np.bool_],
    bbox2: NDArray[np.integer],
    mask2: NDArray[np.bool_],
) -> float:
    """
    Compute Intersection over Union (IoU) between two mask/bbox pairs.

    Parameters
    ----------
    bbox1 : NDArray[np.integer]
        First bounding box coordinates.
    mask1 : NDArray[np.bool_]
        First binary mask.
    bbox2 : NDArray[np.integer]
        Second bounding box coordinates.
    mask2 : NDArray[np.bool_]
        Second binary mask.

    Returns
    -------
    float
        The IoU value between 0 and 1.

    Raises
    ------
    ValueError
        If bboxes have different dimensions or masks have different dimensions.

    Examples
    --------
    Calculate IoU between two overlapping masks:

    >>> import numpy as np
    >>> mask1 = np.array([[True, True], [True, False]])
    >>> bbox1 = np.array([0, 0, 2, 2])
    >>> mask2 = np.array([[True, False], [True, True]])
    >>> bbox2 = np.array([0, 0, 2, 2])
    >>> iou = mask_iou(bbox1, mask1, bbox2, mask2)
    >>> iou  # 2 intersection pixels / 4 union pixels
    0.5
    """
    bbox1 = np.asarray(bbox1, dtype=np.int64)
    bbox2 = np.asarray(bbox2, dtype=np.int64)
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)

    if len(bbox1) != len(bbox2):
        raise ValueError(f"Bboxes must have same length, got {len(bbox1)} and {len(bbox2)}")

    if mask1.ndim != mask2.ndim:
        raise ValueError(f"Masks must have same dimensions, got {mask1.ndim} and {mask2.ndim}")

    return fast_iou_with_bbox(bbox1, bbox2, mask1, mask2)


def mask_intersection(
    bbox1: NDArray[np.integer],
    mask1: NDArray[np.bool_],
    bbox2: NDArray[np.integer],
    mask2: NDArray[np.bool_],
) -> float:
    """
    Compute intersection between two mask/bbox pairs.

    Parameters
    ----------
    bbox1 : NDArray[np.integer]
        First bounding box coordinates.
    mask1 : NDArray[np.bool_]
        First binary mask.
    bbox2 : NDArray[np.integer]
        Second bounding box coordinates.
    mask2 : NDArray[np.bool_]
        Second binary mask.

    Returns
    -------
    float
        The intersection value.

    Raises
    ------
    ValueError
        If bboxes have different dimensions or masks have different dimensions.

    Examples
    --------
    Calculate intersection between two masks:

    >>> import numpy as np
    >>> mask1 = np.array([[True, True], [True, False]])
    >>> bbox1 = np.array([0, 0, 2, 2])
    >>> mask2 = np.array([[True, False], [True, True]])
    >>> bbox2 = np.array([0, 0, 2, 2])
    >>> intersection = mask_intersection(bbox1, mask1, bbox2, mask2)
    >>> intersection  # 2 overlapping pixels
    2.0
    """
    # Ensure inputs are numpy arrays for numba compatibility
    bbox1 = np.asarray(bbox1, dtype=np.int64)
    bbox2 = np.asarray(bbox2, dtype=np.int64)
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)

    if len(bbox1) != len(bbox2):
        raise ValueError(f"Bboxes must have same length, got {len(bbox1)} and {len(bbox2)}")

    if mask1.ndim != mask2.ndim:
        raise ValueError(f"Masks must have same dimensions, got {mask1.ndim} and {mask2.ndim}")

    return fast_intersection_with_bbox(bbox1, bbox2, mask1, mask2)
