"""Utilities for working with segmentation masks stored in geff files.

Segmentation masks are binary, but geff files written by older versions of
tracksdata stored the mask ``data`` buffer as ``uint64`` (see
https://github.com/royerlab/tracksdata/pull/318). That is 8x larger than a
boolean buffer both on disk and, more importantly, when read into memory,
which can cause out-of-memory errors when loading large datasets.

New files store masks as ``bool`` at write time. This module provides a
one-time conversion for legacy files so they can be read cheaply.

The caller names the mask attribute to convert (defaulting to the standard
``DEFAULT_ATTR_KEYS.MASK`` key). Nothing on disk distinguishes a mask from any
other variable-length attribute, so these functions never guess which
attributes are masks — call once per mask key.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import StoreLike

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._logging import LOG

__all__ = ["convert_geff_mask_to_bool", "geff_mask_dtype"]


def geff_mask_dtype(geff_store: StoreLike, mask_key: str = DEFAULT_ATTR_KEYS.MASK) -> np.dtype | None:
    """Return the on-disk dtype of a mask ``data`` buffer in a geff store.

    Parameters
    ----------
    geff_store : StoreLike
        The store or path to the geff group.
    mask_key : str
        Which mask attribute to inspect. Defaults to the standard mask key.

    Returns
    -------
    np.dtype | None
        The dtype of the mask ``data`` array, or ``None`` if there is no such
        variable-length attribute. Compare against ``np.bool_`` to decide
        whether :func:`convert_geff_mask_to_bool` is worth running.
    """
    root = zarr.open_group(geff_store, mode="r")
    try:
        return np.dtype(root[f"nodes/props/{mask_key}/data"].dtype)
    except KeyError:
        return None


def convert_geff_mask_to_bool(
    geff_store: StoreLike,
    mask_key: str = DEFAULT_ATTR_KEYS.MASK,
    *,
    output_path: StoreLike | None = None,
) -> bool:
    """Rewrite one mask attribute's ``data`` buffer to ``bool``.

    Only the mask ``data`` buffer is rewritten; the ``values`` (offset/shape)
    array is left untouched. The buffer is read one native zarr chunk at a time
    so the full non-boolean buffer is never materialized at once. The geff
    metadata dtype is updated to match.

    Parameters
    ----------
    geff_store : StoreLike
        The store or path to the geff group.
    mask_key : str
        The mask attribute to convert. Defaults to the standard mask key. Call
        once per mask attribute if a graph carries more than one.
    output_path : StoreLike | None
        If given, the geff is first copied to this path and the copy is
        converted, leaving the original untouched (safer for shared data). Both
        ``geff_store`` and ``output_path`` must then be filesystem paths.
        If ``None`` (default), the conversion is done in place.

    Returns
    -------
    bool
        ``True`` if a conversion was performed, ``False`` if the mask was
        already boolean.

    Raises
    ------
    KeyError
        If ``mask_key`` is not a variable-length attribute in the geff.
    ValueError
        If the attribute's buffer is neither integer nor boolean (i.e. not a
        mask), to avoid silently corrupting it.
    """
    if output_path is not None:
        geff_store = _copy_geff(geff_store, output_path)

    root = zarr.open_group(geff_store, mode="r+")
    try:
        old = root[f"nodes/props/{mask_key}/data"]
    except KeyError as e:
        raise KeyError(f"{mask_key!r} is not a variable-length (mask) attribute in this geff.") from e

    dtype = np.dtype(old.dtype)
    if dtype == np.bool_:
        return False
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"Refusing to boolify mask {mask_key!r}: buffer dtype {dtype} is not integer.")

    n = int(old.shape[0])
    LOG.info("Converting geff mask %r data (%d elements) from %s to bool", mask_key, n, dtype)

    # Read one native chunk at a time so the full non-boolean buffer is never
    # in memory at once; the resulting boolean buffer is 1/8th its size.
    buf = np.empty(n, dtype=bool)
    step = old.chunks[0]
    for i in range(0, n, step):
        j = min(i + step, n)
        buf[i:j] = np.asarray(old[i:j]).astype(bool)

    _overwrite_array(root[f"nodes/props/{mask_key}"], "data", buf, old.chunks)
    _set_mask_metadata_bool(root, mask_key)
    return True


def _copy_geff(geff_store: StoreLike, output_path: StoreLike) -> Path:
    """Copy a geff directory to ``output_path`` and return the new path."""
    try:
        src = Path(os.fspath(geff_store))
        dst = Path(os.fspath(output_path))
    except TypeError as e:
        raise TypeError("output_path is only supported for filesystem-path geff stores.") from e
    if dst.exists():
        raise FileExistsError(f"output_path already exists: {dst}")
    shutil.copytree(src, dst)
    return dst


def _overwrite_array(parent: zarr.Group, name: str, data: np.ndarray, chunks) -> None:
    """Create (overwriting any existing) a zarr array and write ``data`` to it.

    Works with both zarr v2 and v3 array-creation APIs.
    """
    try:
        # zarr v3
        arr = parent.create_array(name=name, shape=data.shape, chunks=chunks, dtype=data.dtype, overwrite=True)
    except (TypeError, AttributeError):
        # zarr v2
        if name in parent:
            del parent[name]
        arr = parent.create_dataset(name, shape=data.shape, chunks=chunks, dtype=data.dtype, overwrite=True)
    arr[:] = data


def _set_mask_metadata_bool(root: zarr.Group, mask_key: str) -> None:
    """Update the geff node-property metadata so the mask dtype reads as bool."""
    geff_meta = root.attrs.get("geff")
    if not geff_meta:
        return
    node_props = geff_meta.get("node_props_metadata", {})
    mask_meta = node_props.get(mask_key)
    if mask_meta is not None and mask_meta.get("dtype") != "bool":
        mask_meta["dtype"] = "bool"
        # Reassign to persist the nested change back to the store.
        root.attrs["geff"] = geff_meta
