from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import IndexedRXGraph, RustWorkXGraph
from tracksdata.io import convert_geff_prop_dtype, geff_prop_dtype
from tracksdata.io._geff_dtypes import _overwrite_array, _set_prop_metadata_dtype
from tracksdata.nodes._mask import Mask

MASK_KEY = DEFAULT_ATTR_KEYS.MASK


def _make_masked_geff(tmp_path: Path, name: str = "masks.geff") -> tuple[Path, dict[int, np.ndarray]]:
    """Write a small geff with a few masked nodes; return path and node->mask map."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("x", pl.Float64)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))

    masks = [
        np.array([[True, True], [True, False]], dtype=bool),
        np.array([[True, False, True]], dtype=bool),
        np.array([[True], [True], [True]], dtype=bool),
    ]
    bboxes = [
        np.array([6, 6, 8, 8]),
        np.array([0, 0, 1, 3]),
        np.array([2, 5, 5, 6]),
    ]
    node_masks = {}
    for t, (mask, bbox) in enumerate(zip(masks, bboxes, strict=True)):
        node_id = graph.add_node({"t": t, "x": float(t), MASK_KEY: Mask(mask, bbox=bbox), "bbox": bbox})
        node_masks[node_id] = mask

    geff_path = tmp_path / name
    graph.to_geff(geff_store=geff_path)
    return geff_path, node_masks


def test_convert_geff_mask_to_bool_roundtrip(tmp_path: Path) -> None:
    geff_path, node_masks = _make_masked_geff(tmp_path)
    convert_geff_prop_dtype(geff_path, MASK_KEY, np.uint64)

    assert geff_prop_dtype(geff_path, MASK_KEY) == np.uint64

    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_) is True
    assert geff_prop_dtype(geff_path, MASK_KEY) == np.bool_
    # metadata dtype updated too
    root = zarr.open_group(geff_path, mode="r")
    assert root.attrs["geff"]["node_props_metadata"][MASK_KEY]["dtype"] == "bool"

    # masks still load correctly and are boolean
    graph, _ = IndexedRXGraph.from_geff(geff_path)
    for node_id in graph.node_ids():
        loaded = graph.nodes[node_id][MASK_KEY]
        assert loaded.mask.dtype == np.bool_
        np.testing.assert_array_equal(loaded.mask, node_masks[node_id])


def test_convert_is_noop_when_already_bool(tmp_path: Path) -> None:
    geff_path, _ = _make_masked_geff(tmp_path)
    # to_geff already writes bool
    assert geff_prop_dtype(geff_path, MASK_KEY) == np.bool_
    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_) is False


def test_convert_is_idempotent(tmp_path: Path) -> None:
    geff_path, _ = _make_masked_geff(tmp_path)
    convert_geff_prop_dtype(geff_path, MASK_KEY, np.uint64)
    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_) is True
    # second call is a no-op
    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_) is False


def test_missing_mask_key_raises(tmp_path: Path) -> None:
    geff_path, _ = _make_masked_geff(tmp_path)
    assert geff_prop_dtype(geff_path, "nope") is None
    with pytest.raises(KeyError):
        convert_geff_prop_dtype(geff_path, "nope", np.bool_)


def test_output_path_leaves_original_untouched(tmp_path: Path) -> None:
    geff_path, node_masks = _make_masked_geff(tmp_path)
    convert_geff_prop_dtype(geff_path, MASK_KEY, np.uint64)

    out_path = tmp_path / "converted.geff"
    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_, output_path=out_path) is True

    # original is still uint64, copy is bool
    assert geff_prop_dtype(geff_path, MASK_KEY) == np.uint64
    assert geff_prop_dtype(out_path, MASK_KEY) == np.bool_

    graph, _ = IndexedRXGraph.from_geff(out_path)
    for node_id in graph.node_ids():
        np.testing.assert_array_equal(graph.nodes[node_id][MASK_KEY].mask, node_masks[node_id])


def test_explicit_named_mask_key(tmp_path: Path) -> None:
    """A caller with a second mask attribute converts it by naming it explicitly."""
    geff_path, _ = _make_masked_geff(tmp_path)
    # add a second variable-length mask attribute (uint64) by copying the first
    root = zarr.open_group(geff_path, mode="r+")
    props = root["nodes/props"]
    src = props[MASK_KEY]
    dst = props.create_group("nucleus_mask")
    for sub in ("values", "data"):
        arr = src[sub]
        cast = np.uint64 if sub == "data" else arr.dtype
        _overwrite_array(dst, sub, np.asarray(arr[:]).astype(cast), arr.chunks)

    # default call only touches 'mask'
    convert_geff_prop_dtype(geff_path, MASK_KEY, np.uint64)
    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_) is True
    assert geff_prop_dtype(geff_path, "nucleus_mask") == np.uint64  # untouched by default
    # explicit key converts the second mask
    assert convert_geff_prop_dtype(geff_path, "nucleus_mask", np.bool_) is True
    assert geff_prop_dtype(geff_path, "nucleus_mask") == np.bool_


def test_output_path_to_memory_store(tmp_path: Path) -> None:
    """A non-path (in-memory) output_path is supported via a store-agnostic copy."""
    from zarr.storage import MemoryStore

    geff_path, node_masks = _make_masked_geff(tmp_path)
    convert_geff_prop_dtype(geff_path, MASK_KEY, np.uint64)

    out_store = MemoryStore()
    assert convert_geff_prop_dtype(geff_path, MASK_KEY, np.bool_, output_path=out_store) is True

    # original on disk is untouched, in-memory copy is bool and still loads
    assert geff_prop_dtype(geff_path, MASK_KEY) == np.uint64
    assert geff_prop_dtype(out_store, MASK_KEY) == np.bool_
    graph, _ = IndexedRXGraph.from_geff(out_store)
    for node_id in graph.node_ids():
        np.testing.assert_array_equal(graph.nodes[node_id][MASK_KEY].mask, node_masks[node_id])


def test_convert_fixed_length_prop_dtype(tmp_path: Path) -> None:
    """The general converter casts a fixed-length property's ``values`` payload."""
    geff_path, _ = _make_masked_geff(tmp_path)
    # 'x' is a fixed-length float64 node property, stored directly in `values`.
    assert geff_prop_dtype(geff_path, "x") == np.float64
    assert convert_geff_prop_dtype(geff_path, "x", np.float32) is True
    assert geff_prop_dtype(geff_path, "x") == np.float32
    assert convert_geff_prop_dtype(geff_path, "x", np.float32) is False  # no-op


def test_set_prop_metadata_dtype_without_geff_attrs(tmp_path: Path) -> None:
    """Guard: metadata helper is a no-op when the group has no geff attrs."""
    root = zarr.open_group(tmp_path / "empty.zarr", mode="w")
    _set_prop_metadata_dtype(root, "nodes", MASK_KEY, np.dtype(bool))  # should not raise
