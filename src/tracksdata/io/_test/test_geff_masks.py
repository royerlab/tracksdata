from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import IndexedRXGraph, RustWorkXGraph
from tracksdata.io import convert_geff_mask_to_bool, geff_mask_dtype
from tracksdata.io._geff_masks import _overwrite_array, _set_mask_metadata_bool
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


def _downgrade_masks_to_uint64(geff_path: Path, mask_key: str = MASK_KEY) -> None:
    """Simulate a legacy (pre-#318) geff by rewriting a mask data buffer as uint64."""
    root = zarr.open_group(geff_path, mode="r+")
    data = root[f"nodes/props/{mask_key}/data"]
    as_uint64 = np.asarray(data[:]).astype(np.uint64)
    _overwrite_array(root[f"nodes/props/{mask_key}"], "data", as_uint64, data.chunks)
    geff_meta = root.attrs.get("geff")
    if mask_key in geff_meta["node_props_metadata"]:
        geff_meta["node_props_metadata"][mask_key]["dtype"] = "uint64"
        root.attrs["geff"] = geff_meta


def test_convert_geff_mask_to_bool_roundtrip(tmp_path: Path) -> None:
    geff_path, node_masks = _make_masked_geff(tmp_path)
    _downgrade_masks_to_uint64(geff_path)

    assert geff_mask_dtype(geff_path) == np.uint64

    assert convert_geff_mask_to_bool(geff_path) is True
    assert geff_mask_dtype(geff_path) == np.bool_
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
    assert geff_mask_dtype(geff_path) == np.bool_
    assert convert_geff_mask_to_bool(geff_path) is False


def test_convert_is_idempotent(tmp_path: Path) -> None:
    geff_path, _ = _make_masked_geff(tmp_path)
    _downgrade_masks_to_uint64(geff_path)
    assert convert_geff_mask_to_bool(geff_path) is True
    # second call is a no-op
    assert convert_geff_mask_to_bool(geff_path) is False


def test_missing_mask_key_raises(tmp_path: Path) -> None:
    geff_path, _ = _make_masked_geff(tmp_path)
    assert geff_mask_dtype(geff_path, mask_key="nope") is None
    with pytest.raises(KeyError):
        convert_geff_mask_to_bool(geff_path, mask_key="nope")


def test_output_path_leaves_original_untouched(tmp_path: Path) -> None:
    geff_path, node_masks = _make_masked_geff(tmp_path)
    _downgrade_masks_to_uint64(geff_path)

    out_path = tmp_path / "converted.geff"
    assert convert_geff_mask_to_bool(geff_path, output_path=out_path) is True

    # original is still uint64, copy is bool
    assert geff_mask_dtype(geff_path) == np.uint64
    assert geff_mask_dtype(out_path) == np.bool_

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
    _downgrade_masks_to_uint64(geff_path)
    assert convert_geff_mask_to_bool(geff_path) is True
    assert geff_mask_dtype(geff_path, "nucleus_mask") == np.uint64  # untouched by default
    # explicit key converts the second mask
    assert convert_geff_mask_to_bool(geff_path, "nucleus_mask") is True
    assert geff_mask_dtype(geff_path, "nucleus_mask") == np.bool_


def test_non_integer_buffer_refused(tmp_path: Path) -> None:
    """Naming a non-integer variable-length attribute is refused, not corrupted."""
    geff_path, _ = _make_masked_geff(tmp_path)
    root = zarr.open_group(geff_path, mode="r+")
    grp = root["nodes/props"].create_group("floatvar")
    _overwrite_array(grp, "values", np.zeros((3, 2), dtype=np.uint64), (3, 2))
    _overwrite_array(grp, "data", np.array([1.5, 2.5, 3.5], dtype=np.float64), (3,))

    with pytest.raises(ValueError, match="not integer"):
        convert_geff_mask_to_bool(geff_path, "floatvar")
    # left untouched
    assert geff_mask_dtype(geff_path, "floatvar") == np.float64


def test_set_mask_metadata_bool_without_geff_attrs(tmp_path: Path) -> None:
    """Guard: metadata helper is a no-op when the group has no geff attrs."""
    root = zarr.open_group(tmp_path / "empty.zarr", mode="w")
    _set_mask_metadata_bool(root, MASK_KEY)  # should not raise
