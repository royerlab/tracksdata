from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict

import blosc2
import numpy as np
import polars as pl
import skimage.morphology as morph
from numpy.typing import NDArray
from skimage.measure import regionprops

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._iou import fast_intersection_with_bbox, fast_iou_with_bbox
from tracksdata.nodes._generic_nodes import GenericFuncNodeAttrs

if TYPE_CHECKING:
    from skimage.measure._regionprops import RegionProperties

    from tracksdata.graph._base_graph import BaseGraph


MASK_DATA_FIELD = "data"
"""Name of the struct field holding the compressed binary mask."""


class Mask(TypedDict):
    """
    An individual segmentation mask of a single instance (object).

    Attributes
    ----------
    bbox : NDArray[np.integer]
        The bounding box of the region of interest with shape (2 * ndim,).
        The first ndim elements are the start indices and the last ndim elements are the end indices.
        Equivalent to slicing a numpy array with `[start:end]`.
    mask : NDArray[np.bool_]
        A binary array indicating the pixels that are part of the object (e.g. cell, nucleus, etc.).

    Notes
    -----
    Always construct with keyword arguments. `Mask` is a `TypedDict`, so a positional
    argument is interpreted as `dict(...)` of that argument rather than as a field.

    Examples
    --------
    ```python
    mask = Mask(bbox=np.array([0, 0, 2, 2]), mask=np.array([[True, False], [False, True]]))
    ```
    """

    bbox: NDArray[np.integer]
    mask: NDArray[np.bool_]


def _pack_mask_array(mask: NDArray) -> bytes:
    """Compress a mask array into a blosc2 cframe."""
    mask = np.ascontiguousarray(mask)
    prev_nthreads = blosc2.set_nthreads(1)
    # Bypass blosc2 printing overhead by directly creating a schunk and converting it to cframe,
    # instead of using blosc2.pack_tensor
    schunk = blosc2.SChunk(data=mask)
    dtype = mask.dtype.descr if mask.dtype.kind == "V" else mask.dtype.str
    schunk.vlmeta["__pack_tensor__"] = ("numpy", mask.shape, dtype)
    cframe = schunk.to_cframe()
    blosc2.set_nthreads(prev_nthreads)
    return cframe


def _unpack_mask_array(data: bytes) -> NDArray:
    """Decompress a blosc2 cframe into a mask array."""
    prev_nthreads = blosc2.set_nthreads(1)
    mask = blosc2.unpack_tensor(data)
    blosc2.set_nthreads(prev_nthreads)
    return mask


@lru_cache(maxsize=5)
def _nd_sphere(
    radius: int,
    ndim: int,
) -> NDArray[np.bool_]:
    """
    Get a spherical mask of a given radius and dimension.
    """

    if ndim == 2:
        return morph.disk(radius).astype(bool)

    if ndim == 3:
        return morph.ball(radius).astype(bool)

    raise ValueError(f"Spherical is only implemented for 2D and 3D, got ndim={ndim}")


def _unpack(mask: Mask) -> tuple[NDArray[np.bool_], NDArray[np.int64], int]:
    """
    Split a mask into its binary array, an int64 bbox array and its number of dimensions.

    The bbox is always copied so callers can move/resize it without touching the input mask.
    """
    array = mask["mask"]
    return array, np.array(mask["bbox"], dtype=np.int64), array.ndim


def mask_validate(mask: Mask) -> Mask:
    """
    Check that the bounding box and the binary array of a mask agree.

    `Mask` is a plain `TypedDict`, so it is not validated on construction.
    Use this when a mask comes from an untrusted source.

    Parameters
    ----------
    mask : Mask
        The mask to validate.

    Returns
    -------
    Mask
        An equivalent mask whose `bbox` is an `np.int64` array.

    Raises
    ------
    ValueError
        If the bbox dimension or the bbox size does not match the binary array.
    """
    array, bbox, ndim = _unpack(mask)

    if ndim != bbox.shape[0] // 2:
        raise ValueError(f"Mask dimension {ndim} does not match bbox dimension {bbox.shape[0]} // 2")

    bbox_size = bbox[ndim:] - bbox[:ndim]

    if np.any(array.shape != bbox_size):
        raise ValueError(f"Mask shape {array.shape} does not match bbox size {bbox_size}")

    return Mask(bbox=bbox, mask=array)


def mask_crop(
    mask: Mask,
    image: NDArray,
    shape: tuple[int, ...] | None = None,
) -> NDArray:
    """
    Crop the region of a mask from an image.

    Parameters
    ----------
    mask : Mask
        The mask defining the region to crop.
    image : NDArray
        The image to crop from.
    shape : tuple[int, ...] | None
        The shape of the cropped image. If None, the `bbox` will be used.

    Returns
    -------
    NDArray
        The cropped image.
    """
    _, bbox, ndim = _unpack(mask)

    if shape is None:
        slicing = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))

    else:
        center = (bbox[:ndim] + bbox[ndim:]) // 2
        half_shape = np.asarray(shape) // 2
        start = np.maximum(center - half_shape, 0)
        end = np.minimum(center + shape - half_shape, image.shape)
        slicing = tuple(slice(s, e) for s, e in zip(start, end, strict=True))

    return image[slicing]


def mask_indices(
    mask: Mask,
    offset: NDArray[np.integer] | int = 0,
) -> tuple[NDArray[np.integer], ...]:
    """
    Get the indices of the pixels that are part of the object.

    Parameters
    ----------
    mask : Mask
        The mask to get the indices from.
    offset : NDArray[np.integer] | int, optional
        The offset to add to the indices, should be used with bounding box information.

    Returns
    -------
    tuple[NDArray[np.integer], ...]
        The indices of the pixels that are part of the object.
    """
    array, bbox, ndim = _unpack(mask)

    if isinstance(offset, int):
        offset = np.full(ndim, offset)

    indices = list(np.nonzero(array))

    for i, index in enumerate(indices):
        indices[i] = index + bbox[i] + offset[i]

    return tuple(indices)


def mask_paint_buffer(
    mask: Mask,
    buffer: np.ndarray,
    value: int | float,
    offset: NDArray[np.integer] | int = 0,
) -> None:
    """
    Paint object into a buffer.

    Parameters
    ----------
    mask : Mask
        The mask to paint.
    buffer : np.ndarray
        The buffer to paint inplace.
    value : int | float
        The value to paint the object.
    offset : NDArray[np.integer] | int, optional
        The offset to add to the indices, should be used with bounding box information.
    """
    array, bbox, ndim = _unpack(mask)
    shape = buffer.shape

    if isinstance(offset, int):
        starts = [int(bbox[i]) + offset for i in range(ndim)]
        stops = [int(bbox[i + ndim]) + offset for i in range(ndim)]
    else:
        starts = [int(bbox[i]) + int(offset[i]) for i in range(ndim)]
        stops = [int(bbox[i + ndim]) + int(offset[i]) for i in range(ndim)]

    # fast path: bbox fully inside buffer — no numpy allocations
    if all(starts[i] >= 0 and stops[i] <= shape[i] for i in range(ndim)):
        buffer[tuple(slice(starts[i], stops[i]) for i in range(ndim))][array] = value
        return

    # if bboxes falls outside buffer, clip to buffer bounds
    clipped_start = [max(0, starts[i]) for i in range(ndim)]
    clipped_stop = [min(shape[i], stops[i]) for i in range(ndim)]

    if any(clipped_stop[i] <= clipped_start[i] for i in range(ndim)):
        return

    mask_slicing = tuple(
        slice(clipped_start[i] - starts[i], array.shape[i] - (stops[i] - clipped_stop[i])) for i in range(ndim)
    )
    window = tuple(slice(clipped_start[i], clipped_stop[i]) for i in range(ndim))
    buffer[window][array[mask_slicing]] = value


def mask_iou(mask: Mask, other: Mask) -> float:
    """
    Compute the Intersection over Union (IoU) between two masks
    considering their bounding boxes location.

    Parameters
    ----------
    mask : Mask
        The first mask.
    other : Mask
        The other mask to compute the IoU with.

    Returns
    -------
    float
        The IoU between the two masks.
    """
    return fast_iou_with_bbox(np.asarray(mask["bbox"]), np.asarray(other["bbox"]), mask["mask"], other["mask"])


def mask_intersection(mask: Mask, other: Mask) -> float:
    """
    Compute the intersection between two masks considering their bounding boxes location.

    Parameters
    ----------
    mask : Mask
        The first mask.
    other : Mask
        The other mask to compute the intersection with.

    Returns
    -------
    float
        The intersection between the two masks.
    """
    return fast_intersection_with_bbox(np.asarray(mask["bbox"]), np.asarray(other["bbox"]), mask["mask"], other["mask"])


def mask_union(mask: Mask, other: Mask) -> Mask:
    """
    Compute the union mask between two masks considering their bounding boxes location.

    Parameters
    ----------
    mask : Mask
        The first mask.
    other : Mask
        The other mask to compute the union with.

    Returns
    -------
    Mask
        The union mask between the two masks.
    """
    array, bbox, ndim = _unpack(mask)
    other_array, other_bbox, other_ndim = _unpack(other)

    if ndim != other_ndim:
        raise ValueError(f"Cannot compute union between masks of different dimensions: {ndim} and {other_ndim}.")

    start = bbox[:ndim]
    end = bbox[ndim:]
    other_start = other_bbox[:ndim]
    other_end = other_bbox[ndim:]

    union_start = np.minimum(start, other_start)
    union_end = np.maximum(end, other_end)
    union_shape = union_end - union_start

    union_mask = np.zeros(union_shape, dtype=np.bool_)

    slicing = tuple(slice(s - base, e - base) for s, e, base in zip(start, end, union_start, strict=True))
    other_slicing = tuple(
        slice(s - base, e - base) for s, e, base in zip(other_start, other_end, union_start, strict=True)
    )

    union_mask[slicing] |= array
    union_mask[other_slicing] |= other_array

    return Mask(bbox=np.concatenate([union_start, union_end]), mask=union_mask)


def mask_subtract(mask: Mask, other: Mask) -> Mask:
    """
    Compute the difference between two masks considering their bounding boxes location.

    The bounding box is left untouched, only the pixels shared with `other` are removed.

    Parameters
    ----------
    mask : Mask
        The mask to subtract from.
    other : Mask
        The other mask to compute the difference with.

    Returns
    -------
    Mask
        A new mask with the pixels of `other` removed.
    """
    array, bbox, ndim = _unpack(mask)
    other_array, other_bbox, _ = _unpack(other)

    result = Mask(bbox=bbox.copy(), mask=array.copy())

    if mask_intersection(mask, other) == 0:
        return result

    other_slicing = []
    slicing = []
    for i in range(ndim):
        diff = bbox[i] - other_bbox[i]
        if diff > 0:
            start = None
            other_start = diff
        else:
            start = -diff
            other_start = None

        diff = bbox[i + ndim] - other_bbox[i + ndim]
        if diff > 0:
            end = -diff
            other_end = None
        elif diff < 0:
            end = None
            other_end = diff
        else:
            end = None
            other_end = None

        slicing.append(slice(start, end))
        other_slicing.append(slice(other_start, other_end))

    result["mask"][tuple(slicing)] &= ~other_array[tuple(other_slicing)]
    return result


def _crop_overhang(mask: Mask, image_shape: tuple[int, ...]) -> Mask:
    """
    Crop regions of the mask that are outside the image.
    This is used to fix bounding box and mask after changes to the mask.

    Parameters
    ----------
    mask : Mask
        The mask to crop.
    image_shape : tuple[int, ...]
        The shape of the image.

    Returns
    -------
    Mask
        The cropped mask.
    """
    array, bbox, ndim = _unpack(mask)

    left_overhang = np.maximum(0, -bbox[:ndim])
    bbox[:ndim] += left_overhang

    right_overhang = np.maximum(0, bbox[ndim:] - image_shape)
    bbox[ndim:] -= right_overhang

    slicing = tuple(slice(s, -e if e > 0 else None) for s, e in zip(left_overhang, right_overhang, strict=True))

    # validating bbox and mask shape agree after cropping
    return mask_validate(Mask(bbox=bbox, mask=array[slicing]))


def mask_dilate(mask: Mask, radius: int, image_shape: tuple[int, ...] | None = None) -> Mask:
    """
    Dilate a mask by a given radius.

    Parameters
    ----------
    mask : Mask
        The mask to dilate.
    radius : int
        The radius of the dilation.
    image_shape : tuple[int, ...] | None
        The shape of the image.
        When provided handles regions outside the image.

    Returns
    -------
    Mask
        The dilated mask.
    """
    array, bbox, ndim = _unpack(mask)

    new_array = np.pad(array, radius, mode="constant", constant_values=False)

    morph.dilation(
        new_array,
        _nd_sphere(radius, new_array.ndim),
        mode="constant",
        cval=False,
        out=new_array,
    )

    bbox[:ndim] -= radius
    bbox[ndim:] += radius

    dilated = Mask(bbox=bbox, mask=new_array)

    if image_shape is not None:
        dilated = _crop_overhang(dilated, image_shape)

    return dilated


def mask_move(
    mask: Mask,
    offset: NDArray[np.integer],
    image_shape: tuple[int, ...] | None = None,
) -> Mask:
    """
    Move a mask by a given offset.

    Parameters
    ----------
    mask : Mask
        The mask to move.
    offset : NDArray[np.integer]
        The offset to move the mask by.
    image_shape : tuple[int, ...] | None
        The shape of the image.
        When provided handles regions outside the image.

    Returns
    -------
    Mask
        The moved mask.
    """
    array, bbox, ndim = _unpack(mask)

    bbox[:ndim] += offset
    bbox[ndim:] += offset

    moved = Mask(bbox=bbox, mask=array)

    if image_shape is not None:
        moved = _crop_overhang(moved, image_shape)

    return moved


def mask_regionprops(mask: Mask, **kwargs) -> "RegionProperties":
    """
    Compute scikit-image regionprops for a mask.

    The computation is aware of the mask bounding box, so coordinate-based
    properties (e.g. centroid, coords) are returned in absolute
    image coordinates.

    IMPORTANT: When providing an intensity image it should match the bounding box shape, not the actual image shape.

    Parameters
    ----------
    mask : Mask
        The mask to compute the properties of.
    **kwargs : dict
        Keyword arguments to pass to regionprops.
    """
    array, bbox, ndim = _unpack(mask)

    props = regionprops(
        array.astype(np.uint16),
        cache=True,
        offset=tuple(bbox[:ndim]),
        **kwargs,
    )

    if len(props) != 1:
        raise ValueError("Expected a single region in mask to compute regionprops.")

    return props[0]


def mask_size(mask: Mask) -> int:
    """
    Get the number of pixels that are part of the object.

    Parameters
    ----------
    mask : Mask
        The mask to measure.

    Returns
    -------
    int
        The number of pixels that are part of the object.
    """
    return mask["mask"].sum()


def mask_equal(mask: Mask, other: Mask) -> bool:
    """
    Check whether two masks have the same bounding box and binary array.

    `Mask` holds numpy arrays, so `==` cannot be used to compare two masks.

    Parameters
    ----------
    mask : Mask
        The first mask.
    other : Mask
        The mask to compare with.

    Returns
    -------
    bool
        Whether the two masks are equal.
    """
    return np.array_equal(mask["bbox"], other["bbox"]) and np.array_equal(mask["mask"], other["mask"])


def mask_from_coordinates(
    center: NDArray,
    radius: int,
    image_shape: tuple[int, ...] | None = None,
) -> Mask:
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
    array = _nd_sphere(radius, len(center))
    center = np.round(center).astype(int)

    start = center - np.asarray(array.shape) // 2
    end = start + array.shape

    if image_shape is None:
        bbox = np.concatenate([start, end])
    else:
        processed_start = np.maximum(start, 0)
        processed_end = np.minimum(end, image_shape)

        start_overhang = processed_start - start
        end_overhang = end - processed_end

        array = array[tuple(slice(s, -e if e > 0 else None) for s, e in zip(start_overhang, end_overhang, strict=True))]

        bbox = np.concatenate([processed_start, processed_end])

    return Mask(bbox=bbox, mask=array)


def mask_bbox_struct_fields(ndim: int) -> list[str]:
    """
    Names of the bounding box fields of the mask struct attribute.

    Fields follow the ``bbox`` layout (start indices then end indices),
    named after the (z), y, x axis convention, e.g. for 2D:
    ``["min_y", "min_x", "max_y", "max_x"]``.

    Parameters
    ----------
    ndim : int
        The number of spatial dimensions (1 to 3).

    Returns
    -------
    list[str]
        The bounding box field names.
    """
    if ndim < 1 or ndim > 3:
        raise ValueError(f"Mask struct attributes are only supported for 1D to 3D masks, got ndim={ndim}")
    axes = "zyx"[-ndim:]
    return [f"min_{a}" for a in axes] + [f"max_{a}" for a in axes]


def mask_struct_dtype(ndim: int) -> pl.Struct:
    """
    Polars struct dtype used to store a `Mask` as a struct attribute.

    Bounding box coordinates are scalar integer fields so backends can
    filter on them natively (e.g. `NodeAttr("mask").struct.field("min_y") > 5`),
    while the binary mask is stored blosc2-compressed in the ``data`` field.

    Parameters
    ----------
    ndim : int
        The number of spatial dimensions (1 to 3).

    Returns
    -------
    pl.Struct
        The struct dtype, e.g. for 2D:
        `pl.Struct({"min_y": Int64, "min_x": Int64, "max_y": Int64, "max_x": Int64, "data": Binary})`.
    """
    fields = dict.fromkeys(mask_bbox_struct_fields(ndim), pl.Int64)
    fields[MASK_DATA_FIELD] = pl.Binary
    return pl.Struct(fields)


def mask_to_struct(mask: Mask) -> dict[str, Any]:
    """
    Convert a mask to a dict matching [mask_struct_dtype][tracksdata.nodes.mask_struct_dtype].

    Parameters
    ----------
    mask : Mask
        The mask to convert.

    Returns
    -------
    dict[str, Any]
        Scalar bounding box fields plus the blosc2-compressed mask under ``"data"``.
    """
    array, bbox, ndim = _unpack(mask)
    fields = mask_bbox_struct_fields(ndim)
    value: dict[str, Any] = {f: int(b) for f, b in zip(fields, bbox, strict=True)}
    value[MASK_DATA_FIELD] = _pack_mask_array(array)
    return value


def mask_from_struct(value: dict[str, Any]) -> Mask:
    """
    Reconstruct a mask from a struct attribute value.

    Parameters
    ----------
    value : dict[str, Any]
        A dict as produced by [mask_to_struct][tracksdata.nodes.mask_to_struct].

    Returns
    -------
    Mask
        The reconstructed mask.
    """
    array = _unpack_mask_array(value[MASK_DATA_FIELD])
    fields = mask_bbox_struct_fields(array.ndim)
    bbox = np.asarray([value[f] for f in fields], dtype=np.int64)
    return Mask(bbox=bbox, mask=array)


def _decode_mask(value: Mask | dict[str, Any]) -> Mask:
    """
    Decode a single mask attribute value.

    Struct values (as returned by graph backends for struct mask attributes)
    carry the compressed array under ``MASK_DATA_FIELD``; `Mask` values are
    returned unchanged. Prefer [masks_from_column][tracksdata.nodes.masks_from_column]
    when the column dtype is available.
    """
    if MASK_DATA_FIELD in value:
        return mask_from_struct(value)
    return value


def masks_from_column(column: pl.Series) -> list[Mask]:
    """
    Decode a mask attribute column into `Mask` values.

    Dispatches on the column dtype, so it handles both struct mask attributes
    (see [mask_struct_dtype][tracksdata.nodes.mask_struct_dtype]) and `pl.Object`
    columns holding `Mask` values directly.

    Parameters
    ----------
    column : pl.Series
        The mask attribute column.

    Returns
    -------
    list[Mask]
        The decoded masks.
    """
    if isinstance(column.dtype, pl.Struct):
        return [mask_from_struct(v) for v in column]

    if column.dtype == pl.Object:
        return column.to_list()

    raise TypeError(f"Cannot interpret a '{column.dtype}' column as masks.")


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
        If not provided, DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X will be used.
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
            default_columns = [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
            attr_keys = default_columns[-len(image_shape) :]

        if len(attr_keys) != len(image_shape):
            raise ValueError(
                f"Expected image shape {image_shape} to have the same number of dimensions as attr_keys '{attr_keys}'."
            )

        self._image_shape = image_shape

        super().__init__(
            func=lambda **kwargs: mask_to_struct(
                mask_from_coordinates(
                    center=np.asarray(list(kwargs.values())),
                    radius=radius,
                    image_shape=image_shape,
                )
            ),
            output_key=output_key,
            attr_keys=attr_keys,
            batch_size=0,
        )

    def _init_node_attrs(self, graph: "BaseGraph") -> None:
        """
        Validate that the output key exists in the graph.
        """
        if self.output_key not in graph.node_attr_keys():
            graph.add_node_attr_key(self.output_key, mask_struct_dtype(len(self._image_shape)))
