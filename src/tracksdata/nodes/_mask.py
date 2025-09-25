from collections.abc import Sequence
from functools import cached_property, lru_cache
from typing import Any

import blosc2
import numpy as np
import skimage.morphology as morph
from numpy.typing import ArrayLike, NDArray

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._iou import fast_intersection_with_bbox, fast_iou_with_bbox
from tracksdata.nodes._generic_nodes import GenericFuncNodeAttrs


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
        prev_nthreads = blosc2.set_nthreads(1)
        data_dict["_mask"] = blosc2.pack_array2(self._mask)
        blosc2.set_nthreads(prev_nthreads)
        return data_dict

    def __setstate__(self, state: dict) -> None:
        prev_nthreads = blosc2.set_nthreads(1)
        state["_mask"] = blosc2.unpack_array2(state["_mask"])
        blosc2.set_nthreads(prev_nthreads)
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
            center = (self._bbox[: self._mask.ndim] + self._bbox[self._mask.ndim :]) // 2
            half_shape = np.asarray(shape) // 2
            start = np.maximum(center - half_shape, 0)
            end = np.minimum(center + half_shape, image.shape)
            slicing = tuple(slice(s, e) for s, e in zip(start, end, strict=True))

        return image[slicing]

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
        if isinstance(offset, int):
            offset = np.full(self._mask.ndim, offset)

        window = tuple(
            slice(i + o, j + o)
            for i, j, o in zip(self._bbox[: self._mask.ndim], self._bbox[self._mask.ndim :], offset, strict=True)
        )
        buffer[window][self._mask] = value

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
        center = np.round(center).astype(int)

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Mask):
            return False
        return np.array_equal(self.bbox, other.bbox) and np.array_equal(self.mask, other.mask)


class MaskDiskAttrs(GenericFuncNodeAttrs):
    """
    Operator to create a disk mask for each node.

    Masks are created in space, so temporal information should not be provided.

    Parameters
    ----------
    radius : int
        The radius of the mask.
    image_shape : tuple[int, ...]
        The shape of the image, must match the number of  of the attr_keys.
    attr_keys : Sequence[str] | None
        The attributes for the center of the mask.
        If not provided, "z", "y", "x" will be used.
    output_key : str
        The key of the attribute to store the mask.
    """

    def __init__(
        self,
        radius: int,
        image_shape: tuple[int, ...],
        attr_keys: Sequence[str] | None = None,
        output_key: str = DEFAULT_ATTR_KEYS.MASK,
    ):
        if attr_keys is None:
            default_columns = ["z", "y", "x"]
            attr_keys = default_columns[-len(image_shape) :]

        if len(attr_keys) != len(image_shape):
            raise ValueError(
                f"Expected image shape {image_shape} to have the same number of dimensions as attr_keys '{attr_keys}'."
            )

        super().__init__(
            func=lambda **kwargs: Mask.from_coordinates(
                center=np.asarray(list(kwargs.values())),
                radius=radius,
                image_shape=image_shape,
            ),
            output_key=output_key,
            attr_keys=attr_keys,
            default_value=None,
            batch_size=0,
        )
