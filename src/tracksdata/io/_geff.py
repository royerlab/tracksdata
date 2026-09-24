"""Utilities for inspecting and repairing geff datasets without loading the graph.

:func:`read_graph_metadata` reads the tracksdata graph metadata of a geff dataset.
`BaseGraph.to_geff` writes the graph metadata (`graph.metadata`) into the geff
metadata extras and `BaseGraph.from_geff` hoists it back onto the graph, but callers
that need a value *before* they have a graph object -- for example the ``shape`` of
the dense segmentation, which is required to construct a `GraphArrayView` -- can use
neither. This closes that gap so downstream libraries do not have to know where
tracksdata stores the extras.

:func:`geff_prop_dtype` and :func:`convert_geff_prop_dtype` inspect and convert the
on-disk dtype of a property. The motivating case is segmentation masks: they are
binary, but geff files written by older versions of tracksdata stored the mask
``data`` buffer as ``uint64`` (see
https://github.com/royerlab/tracksdata/pull/318). That is 8x larger than a boolean
buffer both on disk and, more importantly, when read into memory, which can cause
out-of-memory errors when loading large datasets. New files store masks as ``bool``
at write time, so :func:`convert_geff_prop_dtype` provides a one-time fix for legacy
files. The helpers are not mask-specific: they read and rewrite the payload dtype of
any geff property (node or edge, fixed- or variable-length). The caller names the
property to act on.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from geff_spec import GeffMetadata
from zarr.storage import StoreLike

from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG

__all__ = [
    "append_graph_metadata",
    "convert_geff_prop_dtype",
    "geff_prop_dtype",
    "read_graph_metadata",
    "remove_graph_metadata",
]

_EXTRA_KEY = "tracksdata"


def read_graph_metadata(source: StoreLike | GeffMetadata) -> dict[str, Any]:
    """
    Read the tracksdata graph metadata of a geff dataset without loading the graph.

    Returns the same metadata that `graph.metadata` would hold after
    `BaseGraph.from_geff`, minus the `geff` key. Note that the values went through a
    JSON round-trip, so tuples come back as lists.

    Parameters
    ----------
    source : StoreLike | GeffMetadata
        The store or path of the geff dataset, or an already parsed `GeffMetadata`.

    Returns
    -------
    dict[str, Any]
        The graph metadata, empty if the dataset was not written by tracksdata.

    Examples
    --------
    ```python
    graph.metadata["shape"] = (5, 100, 100)
    graph.to_geff("tracks.geff")

    shape = read_graph_metadata("tracks.geff")["shape"]  # [5, 100, 100]
    ```
    """
    if not isinstance(source, GeffMetadata):
        source = GeffMetadata.read(source)

    metadata = source.extra.get(_EXTRA_KEY, {})

    return {k: v for k, v in metadata.items() if not BaseGraph._is_private_metadata_key(k)}


def append_graph_metadata(store: StoreLike, **kwargs: Any) -> None:
    """
    Append tracksdata graph metadata into an existing geff store, without loading the graph.

    Counterpart to :func:`read_graph_metadata`. Merges ``kwargs`` into the store's
    existing `extra["tracksdata"]` entry, so previously written keys are preserved
    unless overwritten. Private metadata keys (starting with
    `BaseGraph._PRIVATE_METADATA_PREFIX`) are rejected, matching `graph.metadata`.

    Parameters
    ----------
    store : StoreLike
        The store or path of the geff dataset.
    **kwargs : Any
        The metadata entries to write, e.g. `shape=(5, 100, 100)`.

    Examples
    --------
    ```python
    append_graph_metadata("tracks.geff", shape=(5, 100, 100))
    read_graph_metadata("tracks.geff")["shape"]  # [5, 100, 100]
    ```
    """
    BaseGraph._validate_metadata_keys(kwargs.keys(), is_public=True)

    metadata = GeffMetadata.read(store)
    extra = dict(metadata.extra)
    extra[_EXTRA_KEY] = {**extra.get(_EXTRA_KEY, {}), **kwargs}
    metadata.extra = extra
    metadata.write(store)


def remove_graph_metadata(store: StoreLike, key: str) -> None:
    """
    Remove a tracksdata graph metadata entry from an existing geff store, without loading the graph.

    Counterpart to :func:`read_graph_metadata`. A no-op if ``key`` is not present.

    Parameters
    ----------
    store : StoreLike
        The store or path of the geff dataset.
    key : str
        The metadata key to remove.

    Examples
    --------
    ```python
    remove_graph_metadata("tracks.geff", "shape")
    "shape" in read_graph_metadata("tracks.geff")  # False
    ```
    """
    BaseGraph._validate_metadata_key(key, is_public=True)

    metadata = GeffMetadata.read(store)
    extra = dict(metadata.extra)
    tracksdata_extra = dict(extra.get(_EXTRA_KEY, {}))
    tracksdata_extra.pop(key, None)
    extra[_EXTRA_KEY] = tracksdata_extra
    metadata.extra = extra
    metadata.write(store)


def geff_prop_dtype(
    geff_store: StoreLike,
    key: str,
    *,
    node_or_edge: str = "nodes",
) -> np.dtype | None:
    """Return the on-disk payload dtype of a property in a geff store.

    Parameters
    ----------
    geff_store : StoreLike
        The store or path to the geff group.
    key : str
        Which property to inspect.
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
    back to a store-agnostic recursive copy, so in-memory and remote stores work
    too.
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
    """Recursively copy a geff group from one store to another.

    Uses only the public zarr ``Group``/``Array`` API (no store internals), so
    it works across zarr 2.x and 3.x.
    """
    stack = [(zarr.open_group(src, mode="r"), zarr.open_group(dst, mode="a"))]
    while stack:
        src_group, dst_group = stack.pop()
        dst_group.attrs.update(dict(src_group.attrs))
        for name, arr in src_group.arrays():
            _overwrite_array(dst_group, name, arr[...], arr.chunks)
            dst_group[name].attrs.update(dict(arr.attrs))
        for name, sub in src_group.groups():
            stack.append((sub, dst_group.require_group(name)))


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
