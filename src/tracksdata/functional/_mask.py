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
    """
    ndim = len(bbox) // 2

    if shape is None:
        slicing = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))
    else:
        center = (bbox[:ndim] + bbox[ndim:]) // 2
        half_shape = np.asarray(shape) // 2
        start = np.maximum(center - half_shape, 0)
        end = np.minimum(center + half_shape, image.shape)
        slicing = tuple(slice(s, e) for s, e in zip(start, end, strict=True))

    return image[slicing]


def mask_indices(
    mask: NDArray[np.bool_],
    bbox: NDArray[np.integer],
    offset: NDArray[np.integer] | int = 0,
) -> tuple[NDArray[np.integer], ...]:
    """
    Get the indices of pixels that are part of the mask in global coordinates.

    Parameters
    ----------
    mask : NDArray[np.bool_]
        Binary mask indicating valid pixels.
    bbox : NDArray[np.integer]
        The bounding box coordinates with shape (2 * ndim,).
    offset : NDArray[np.integer] | int, optional
        Additional offset to add to the indices.

    Returns
    -------
    tuple[NDArray[np.integer], ...]
        The indices of pixels that are part of the mask.
    """
    ndim = len(bbox) // 2

    if isinstance(offset, int):
        offset = np.full(ndim, offset)

    indices = list(np.nonzero(mask))

    for i, index in enumerate(indices):
        indices[i] = index + bbox[i] + offset[i]

    return tuple(indices)


def paint_mask_to_buffer(
    buffer: np.ndarray,
    mask: NDArray[np.bool_],
    bbox: NDArray[np.integer],
    value: int | float,
    offset: NDArray[np.integer] | int = 0,
) -> None:
    """
    Paint mask pixels into a buffer.

    Parameters
    ----------
    buffer : np.ndarray
        The buffer to paint into (modified in place).
    mask : NDArray[np.bool_]
        Binary mask indicating pixels to paint.
    bbox : NDArray[np.integer]
        The bounding box coordinates.
    value : int | float
        The value to paint.
    offset : NDArray[np.integer] | int, optional
        Additional offset to add to the indices.
    """
    indices = mask_indices(mask, bbox, offset)
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
    """
    bbox1 = np.asarray(bbox1, dtype=np.int64)
    bbox2 = np.asarray(bbox2, dtype=np.int64)
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)

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
    """
    # Ensure inputs are numpy arrays for numba compatibility
    bbox1 = np.asarray(bbox1, dtype=np.int64)
    bbox2 = np.asarray(bbox2, dtype=np.int64)
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)

    return fast_intersection_with_bbox(bbox1, bbox2, mask1, mask2)
