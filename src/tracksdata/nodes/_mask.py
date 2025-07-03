from functools import cached_property, lru_cache

import blosc2
import numpy as np
import skimage.morphology as morph
from numba import njit
from numpy.typing import ArrayLike, NDArray

from tracksdata.functional._iou import fast_intersection_with_bbox, fast_iou_with_bbox


@njit
def bbox_interpolation_offset(
    tgt_bbox: np.ndarray,
    src_bbox: np.ndarray,
    w: float,
) -> np.ndarray:
    """
    Interpolate the bounding box between two masks.
    The reference is the target mask and w is relative distance to it:
    ```python
    (target_value - new_value) / (target_value - source_value) = w
    ```

    Parameters
    ----------
    tgt_bbox : np.ndarray
        The target bounding box.
    src_bbox : np.ndarray
        The source bounding box.
    w : float
        The weight of the interpolation.

    Returns
    -------
    np.ndarray
        The offset to add to the bounding box.
    """
    if w < 0 or w > 1:
        raise ValueError(f"w = {w} is not between 0 and 1")

    ndim = tgt_bbox.shape[0] // 2
    tgt_center = tgt_bbox[ndim:] - tgt_bbox[:ndim] // 2
    src_center = src_bbox[ndim:] - src_bbox[:ndim] // 2
    signed_dist = tgt_center - src_center
    offset = -np.round((1 - w) * signed_dist).astype(np.int32)

    for i in range(ndim):
        if offset[i] > 0:
            new_value = tgt_bbox[ndim + i] - offset[i]
            dist_to_border = min(new_value - tgt_bbox[ndim + i], 0)
            offset[i] += dist_to_border
        else:
            new_value = tgt_bbox[i] + offset[i]
            dist_to_border = max(tgt_bbox[i] - new_value, 0)
            offset[i] += dist_to_border

    return offset


@lru_cache(maxsize=5)
def _spherical_mask(
    radius: int,
    ndim: int,
) -> NDArray[np.bool_]:
    """
    Get a spherical mask of a given radius and dimension.
    """

    if ndim == 2:
        return morph.disk(radius)

    if ndim == 3:
        return morph.ball(radius)

    raise ValueError(f"Spherical is only implemented for 2D and 3D, got ndim={ndim}")


class Mask:
    """
    Object used to store an individual segmentation mask of a single instance (object)

    Parameters
    ----------
    mask : NDArray[np.bool_]
        A binary indicating the pixels that are part of the object (e.g. cell, nucleus, etc.).
    bbox : np.ndarray
        The bounding box of the region of interest with shape (2 * ndim,).
        The first ndim elements are the start indices and the last ndim elements are the end indices.
        Equivalent to slicing a numpy array with `[start:end]`.
    Examples
    --------
    ```python
    mask = Mask(mask=np.array([[True, False], [False, True]]), bbox=np.array([0, 0, 2, 2]))
    ```
    """

    def __init__(
        self,
        mask: NDArray[np.bool_],
        bbox: ArrayLike,
    ):
        self._mask = mask
        self.bbox = bbox

    def __getstate__(self) -> dict:
        data_dict = self.__dict__.copy()
        data_dict["_mask"] = blosc2.pack_array(self._mask)
        return data_dict

    def __setstate__(self, state: dict) -> None:
        state["_mask"] = blosc2.unpack_array(state["_mask"])
        self.__dict__.update(state)

    @property
    def mask(self) -> NDArray[np.bool_]:
        return self._mask

    @property
    def bbox(self) -> NDArray[np.int64]:
        return self._bbox

    @bbox.setter
    def bbox(self, bbox: ArrayLike) -> None:
        bbox = np.asarray(bbox, dtype=np.int64)

        if self._mask.ndim != bbox.shape[0] // 2:
            raise ValueError(f"Mask dimension {self._mask.ndim} does not match bbox dimension {bbox.shape[0]} // 2")

        bbox_size = bbox[self._mask.ndim :] - bbox[: self._mask.ndim]

        if np.any(self._mask.shape != bbox_size):
            raise ValueError(f"Mask shape {self._mask.shape} does not match bbox size {bbox_size}")

        self._bbox: NDArray[np.int64] = bbox

    def crop(
        self,
        image: NDArray,
        shape: tuple[int, ...] | None = None,
    ) -> NDArray:
        """
        Crop the mask from an image.

        Parameters
        ----------
        image : NDArray
            The image to crop from.
        shape : tuple[int, ...] | None
            The shape of the cropped image. If None, the `bbox` will be used.

        Returns
        -------
        NDArray
            The cropped image.
        """
        if shape is None:
            ndim = self._mask.ndim
            slicing = tuple(slice(self._bbox[i], self._bbox[i + ndim]) for i in range(ndim))

        else:
            center = self.bbox_center()
            half_shape = np.asarray(shape) // 2
            start = np.maximum(center - half_shape, 0)
            end = np.minimum(center + half_shape, image.shape)
            slicing = tuple(slice(s, e) for s, e in zip(start, end, strict=True))

        return image[slicing]

    def bbox_center(self) -> NDArray[np.integer]:
        """
        Get the center of the bounding box.
        """
        return (self._bbox[: self._mask.ndim] + self._bbox[self._mask.ndim :]) // 2

    def mask_indices(
        self,
        offset: NDArray[np.integer] | int = 0,
    ) -> tuple[NDArray[np.integer], ...]:
        """
        Get the indices of the pixels that are part of the object.

        Parameters
        ----------
        offset : NDArray[np.integer] | int, optional
            The offset to add to the indices, should be used with bounding box information.

        Returns
        -------
        tuple[NDArray[np.integer], ...]
            The indices of the pixels that are part of the object.
        """
        if isinstance(offset, int):
            offset = np.full(self._mask.ndim, offset)

        indices = list(np.nonzero(self._mask))

        for i, index in enumerate(indices):
            indices[i] = index + self._bbox[i] + offset[i]

        return tuple(indices)

    def paint_buffer(
        self,
        buffer: np.ndarray,
        value: int | float,
        offset: NDArray[np.integer] | int = 0,
    ) -> None:
        """
        Paint object into a buffer.

        Parameters
        ----------
        buffer : np.ndarray
            The buffer to paint inplace.
        value : int | float
            The value to paint the object.
        offset : NDArray[np.integer] | int, optional
            The offset to add to the indices, should be used with bounding box information.
        """
        # TODO: make it zarr and tensorstore compatible
        indices = self.mask_indices(offset)
        buffer[indices] = value

    def iou(self, other: "Mask") -> float:
        """
        Compute the Intersection over Union (IoU) between two masks
        considering their bounding boxes location.

        Parameters
        ----------
        other : Mask
            The other mask to compute the IoU with.

        Returns
        -------
        float
            The IoU between the two masks.
        """
        return fast_iou_with_bbox(self._bbox, other._bbox, self._mask, other._mask)

    def intersection(self, other: "Mask") -> float:
        """
        Compute the intersection between two masks considering their bounding boxes location.

        Parameters
        ----------
        other : Mask
            The other mask to compute the intersection with.

        Returns
        -------
        float
            The intersection between the two masks.
        """
        return fast_intersection_with_bbox(self._bbox, other._bbox, self._mask, other._mask)

    @cached_property
    def size(self) -> int:
        """
        Get the number of pixels that are part of the object.
        """
        return self._mask.sum()

    def __repr__(self) -> str:
        slicing_str = ", ".join(
            f"{i}:{j}"
            for i, j in zip(
                self._bbox[: self._mask.ndim],
                self._bbox[self._mask.ndim :],
                strict=True,
            )
        )
        return f"Mask(bbox=[{slicing_str}])"

    @classmethod
    def from_coordinates(
        cls,
        center: NDArray,
        radius: int,
        image_shape: tuple[int, ...] | None = None,
    ) -> "Mask":
        """
        Create a mask from a center and a radius.
        Regions outside the image are cropped.

        Parameters
        ----------
        center : NDArray
            The center of the mask.
        radius : int
            The radius of the mask.
        image_shape : tuple[int, ...] | None
            The shape of the image.
            When provided crops regions outside the image.

        Returns
        -------
        Mask
            The mask.
        """
        mask = _spherical_mask(radius, len(center))
        center = np.round(center)

        start = center - np.asarray(mask.shape) // 2
        end = start + mask.shape

        if image_shape is None:
            bbox = np.concatenate([start, end])
        else:
            processed_start = np.maximum(start, 0)
            processed_end = np.minimum(end, image_shape)

            start_overhang = processed_start - start
            end_overhang = end - processed_end

            mask = mask[
                tuple(slice(s, -e if e > 0 else None) for s, e in zip(start_overhang, end_overhang, strict=True))
            ]

            bbox = np.concatenate([processed_start, processed_end])

        return cls(mask, bbox)
