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

__all__ = ["convert_geff_prop_dtype", "geff_prop_dtype"]


def geff_prop_dtype(
    geff_store: StoreLike,
    key: str = DEFAULT_ATTR_KEYS.MASK,
    *,
    node_or_edge: str = "nodes",
) -> np.dtype | None:
    """Return the on-disk payload dtype of a property in a geff store.

    Parameters
    ----------
    geff_store : StoreLike
        The store or path to the geff group.
    key : str
        Which property to inspect. Defaults to the standard mask key.
    node_or_edge : str
        Whether ``key`` is a node (``"nodes"``, default) or edge (``"edges"``)
        property.

    Returns
    -------
    np.dtype | None
        The dtype of the property's payload buffer (``data`` for variable-length
        properties such as masks, otherwise ``values``), or ``None`` if there is
        no such property. Compare against ``np.bool_`` to decide whether
        :func:`convert_geff_prop_dtype` is worth running.
    """
    if node_or_edge not in ("nodes", "edges"):
        raise ValueError(f"node_or_edge must be 'nodes' or 'edges', got {node_or_edge!r}.")
    root = zarr.open_group(geff_store, mode="r")
    try:
        group = root[f"{node_or_edge}/props/{key}"]
        payload = "data" if "data" in group else "values"
        return np.dtype(group[payload].dtype)
    except KeyError:
        return None


def convert_geff_prop_dtype(
    geff_store: StoreLike,
    key: str,
    dtype: np.typing.DTypeLike,
    *,
    node_or_edge: str = "nodes",
    output_path: StoreLike | None = None,
) -> bool:
    """Rewrite one property's payload buffer to ``dtype``.

    Only the payload buffer is rewritten (``data`` for variable-length
    properties such as masks, otherwise ``values``); for variable-length
    properties the ``values`` offset/shape array is left untouched. The buffer
    is read one native zarr chunk at a time so the full buffer is never
    materialized at once. The geff metadata dtype is updated to match.

    Parameters
    ----------
    geff_store : StoreLike
        The store or path to the geff group.
    key : str
        The property to convert.
    dtype : np.typing.DTypeLike
        The target dtype for the payload buffer.
    node_or_edge : str
        Whether ``key`` is a node (``"nodes"``, default) or edge (``"edges"``)
        property.
    output_path : StoreLike | None
        If given, the geff is first copied to this store/path and the copy is
        converted, leaving the original untouched (safer for shared data). If
        ``None`` (default), the conversion is done in place.

    Returns
    -------
    bool
        ``True`` if a conversion was performed, ``False`` if the payload already
        had ``dtype``.

    Raises
    ------
    KeyError
        If ``key`` is not a property of ``node_or_edge`` in the geff.
    """
    if node_or_edge not in ("nodes", "edges"):
        raise ValueError(f"node_or_edge must be 'nodes' or 'edges', got {node_or_edge!r}.")
    dtype = np.dtype(dtype)

    if output_path is not None:
        geff_store = _copy_geff(geff_store, output_path)

    root = zarr.open_group(geff_store, mode="r+")
    try:
        group = root[f"{node_or_edge}/props/{key}"]
    except KeyError as e:
        raise KeyError(f"{key!r} is not a {node_or_edge[:-1]} property in this geff.") from e

    payload = "data" if "data" in group else "values"
    old = group[payload]
    if np.dtype(old.dtype) == dtype:
        return False

    n = int(old.shape[0])
    LOG.info("Converting geff %s %r %s (%d elements) from %s to %s", node_or_edge, key, payload, n, old.dtype, dtype)

    # Read one native chunk (along the first axis) at a time so the full buffer
    # is never in memory at once.
    buf = np.empty(old.shape, dtype=dtype)
    step = old.chunks[0]
    for i in range(0, n, step):
        j = min(i + step, n)
        buf[i:j] = np.asarray(old[i:j]).astype(dtype)

    _overwrite_array(group, payload, buf, old.chunks)
    _set_prop_metadata_dtype(root, node_or_edge, key, dtype)
    return True


def _copy_geff(geff_store: StoreLike, output_path: StoreLike) -> StoreLike:
    """Copy a geff store to ``output_path`` and return the destination.

    Uses a fast filesystem copy when both ends are local paths; otherwise falls
    back to a store-agnostic key-by-key copy (zarr v3 has no working
    ``copy_store``), so in-memory and remote stores work too.
    """
    src = _as_path(geff_store)
    dst = _as_path(output_path)
    if src is not None and dst is not None:
        if dst.exists():
            raise FileExistsError(f"output_path already exists: {dst}")
        shutil.copytree(src, dst)
        return dst

    # Fallback to a store-agnostic copy for in-memory or remote stores.
    _copy_store_contents(geff_store, output_path)
    return output_path


def _as_path(store: StoreLike) -> Path | None:
    """Return ``store`` as a filesystem ``Path`` if it is one, else ``None``."""
    try:
        return Path(os.fspath(store))
    except TypeError:
        return None


def _copy_store_contents(src: StoreLike, dst: StoreLike) -> None:
    """Copy every key from one geff store to another, key by key."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    src_store = zarr.open_group(src, mode="r").store
    dst_store = zarr.open_group(dst, mode="a").store
    prototype = default_buffer_prototype()

    async def _run() -> None:
        async for k in src_store.list():
            value = await src_store.get(k, prototype=prototype)
            if value is not None:
                await dst_store.set(k, value)

    sync(_run())


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


def _set_prop_metadata_dtype(root: zarr.Group, node_or_edge: str, key: str, dtype: np.dtype) -> None:
    """Update the geff property metadata so ``key``'s dtype reads as ``dtype``."""
    geff_meta = root.attrs.get("geff")
    if not geff_meta:
        return
    meta_key = "node_props_metadata" if node_or_edge == "nodes" else "edge_props_metadata"
    props = geff_meta.get(meta_key, {})
    prop_meta = props.get(key)
    if prop_meta is not None and prop_meta.get("dtype") != dtype.name:
        prop_meta["dtype"] = dtype.name
        # Reassign to persist the nested change back to the store.
        root.attrs["geff"] = geff_meta
