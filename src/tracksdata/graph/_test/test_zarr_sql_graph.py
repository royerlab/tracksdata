import pickle
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import geff
import numpy as np
import polars as pl
import pytest
import zarr

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph, ZarrSQLGraph
from tracksdata.nodes._mask import Mask
from tracksdata.utils._dtypes import AttrSchema


def _write_geff(path: Path, *, zarr_format: Literal[2, 3] = 3) -> tuple[list[int], Path]:
    graph = RustWorkXGraph()
    graph.add_node_attr_key(AttrSchema("score", pl.Float32))
    graph.add_node_attr_key(AttrSchema("label", pl.Int16))
    graph.add_node_attr_key(AttrSchema("is_even", pl.Boolean))
    graph.add_node_attr_key(AttrSchema(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4)))
    graph.add_edge_attr_key(AttrSchema("weight", pl.Float32))

    node_ids = [
        graph.add_node(
            {
                DEFAULT_ATTR_KEYS.T: time,
                "score": np.float32(time + 0.5),
                "label": np.int16(time + 10),
                "is_even": time % 2 == 0,
                DEFAULT_ATTR_KEYS.BBOX: np.asarray([time, time + 1, time + 2, time + 3]),
            }
        )
        for time in range(3)
    ]
    graph.add_edge(node_ids[0], node_ids[1], {"weight": np.float32(0.25)})
    graph.add_edge(node_ids[1], node_ids[2], {"weight": np.float32(1.25)})
    graph.metadata.update(dataset="demo", shape=[3, 16, 16])
    graph.to_geff(path, zarr_format=zarr_format)
    return node_ids, path


@pytest.fixture
def geff_graph(tmp_path: Path) -> tuple[list[int], Path]:
    return _write_geff(tmp_path / "graph.geff")


def _create_zarr_array(group: Any, name: str, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
    """Create an array through the Zarr v2/v3-compatible API."""
    kwargs: dict[str, Any] = {"data": data}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if hasattr(group, "create_array"):
        group.create_array(name, **kwargs)
    else:
        group.create_dataset(name, shape=data.shape, **kwargs)


def test_constructs_lazily_without_geff_read(
    geff_graph: tuple[list[int], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    node_ids, path = geff_graph

    def fail_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("geff.read must not be used")

    monkeypatch.setattr(geff, "read", fail_read)
    original_getitem = zarr.Array.__getitem__
    selections: list[tuple[object, ...]] = []

    def track_array_reads(array: zarr.Array, selection: tuple[object, ...]) -> Any:
        selections.append(selection)
        return original_getitem(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", track_array_reads)
    graph = ZarrSQLGraph(path)
    assert selections
    assert all(any(isinstance(index, slice) and index.stop == 0 for index in selection) for selection in selections)

    sql = MagicMock(wraps=graph._context.sql)
    monkeypatch.setattr(graph._context, "sql", sql)

    assert graph.node_ids() == node_ids
    assert graph.edge_ids() == [0, 1]
    assert graph.num_nodes() == 3
    assert graph.num_edges() == 2
    assert sql.call_count >= 4


def test_zarr_v2_and_pickling(tmp_path: Path) -> None:
    node_ids, path = _write_geff(tmp_path / "graph-v2.geff", zarr_format=2)

    restored = pickle.loads(pickle.dumps(ZarrSQLGraph(path)))

    assert restored.node_ids() == node_ids
    assert restored.edge_list() == [[node_ids[0], node_ids[1]], [node_ids[1], node_ids[2]]]


def test_scalar_schema_attrs_and_fixed_array(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    graph = ZarrSQLGraph(path)

    assert graph._node_attr_schemas()["score"].dtype == pl.Float32
    assert graph._node_attr_schemas()["label"].dtype == pl.Int16
    assert graph._node_attr_schemas()[DEFAULT_ATTR_KEYS.BBOX].dtype == pl.Array(pl.Int64, 4)
    assert graph._edge_attr_schemas()["weight"].dtype == pl.Float32

    attrs = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "score", "label", DEFAULT_ATTR_KEYS.BBOX])
    assert attrs.schema == {
        DEFAULT_ATTR_KEYS.NODE_ID: pl.UInt64,
        "score": pl.Float32,
        "label": pl.Int16,
        DEFAULT_ATTR_KEYS.BBOX: pl.Array(pl.Int64, 4),
    }
    assert attrs[DEFAULT_ATTR_KEYS.NODE_ID].to_list() == node_ids
    np.testing.assert_array_equal(attrs[DEFAULT_ATTR_KEYS.BBOX][2].to_numpy(), [2, 3, 4, 5])

    edges = graph.edge_attrs(attr_keys=["weight"])
    assert edges.columns == ["weight", "edge_id", "source_id", "target_id"]
    assert edges["weight"].to_list() == pytest.approx([0.25, 1.25])


def test_inconsistent_property_chunks_are_unified(geff_graph: tuple[list[int], Path]) -> None:
    _, path = geff_graph
    root = zarr.open_group(path, mode="a")
    score_group = root["nodes/props/score"]
    values = np.asarray(score_group["values"][:])
    del score_group["values"]
    _create_zarr_array(score_group, "values", values, chunks=(1,))

    attrs = ZarrSQLGraph(path).node_attrs(attr_keys=["score"])

    assert attrs["score"].to_list() == pytest.approx([0.5, 1.5, 2.5])


def test_nullable_float_remains_sql_filterable(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    root = zarr.open_group(path, mode="a")
    missing = np.asarray([False, True, False])
    _create_zarr_array(root["nodes/props/score"], "missing", missing)

    graph = ZarrSQLGraph(path)

    assert graph.node_attrs(attr_keys=["score"])["score"].to_list() == [0.5, None, 2.5]
    assert graph.filter(NodeAttr("score") > 1).node_ids() == [node_ids[2]]


def test_missing_masks_restore_nullable_declared_dtype(geff_graph: tuple[list[int], Path]) -> None:
    _, path = geff_graph
    root = zarr.open_group(path, mode="a")
    missing = np.asarray([False, True, False])
    _create_zarr_array(root["nodes/props/label"], "missing", missing)

    graph = ZarrSQLGraph(path)
    attrs = graph.node_attrs(attr_keys=["label"])

    assert attrs.schema == {"label": pl.Int16}
    assert attrs["label"].to_list() == [10, None, 12]
    with pytest.raises(NotImplementedError, match="Deferred GEFF properties"):
        graph.filter(NodeAttr("label") == 10)


def test_preserves_full_uint64_node_id_range(geff_graph: tuple[list[int], Path]) -> None:
    _, path = geff_graph
    root = zarr.open_group(path, mode="a")
    node_ids = np.asarray([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype=np.uint64)
    root["nodes/ids"][:] = node_ids
    root["edges/ids"][:] = np.asarray([[node_ids[0], node_ids[1]], [node_ids[1], node_ids[2]]])

    graph = ZarrSQLGraph(path)
    expected_ids = node_ids.tolist()

    assert graph.node_ids() == expected_ids
    assert graph.edge_list() == [[expected_ids[0], expected_ids[1]], [expected_ids[1], expected_ids[2]]]
    assert graph.successors(expected_ids[0]) == [expected_ids[1]]
    assert graph._node_attr_schemas()[DEFAULT_ATTR_KEYS.NODE_ID].dtype == pl.UInt64
    assert graph._edge_attr_schemas()[DEFAULT_ATTR_KEYS.EDGE_SOURCE].dtype == pl.UInt64


def test_rejects_undirected_geff(geff_graph: tuple[list[int], Path]) -> None:
    _, path = geff_graph
    root = zarr.open_group(path, mode="a")
    metadata = dict(root.attrs["geff"])
    metadata["directed"] = False
    root.attrs["geff"] = metadata

    with pytest.raises(ValueError, match="only supports directed"):
        ZarrSQLGraph(path)


def test_copy_reopens_the_same_store(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    graph = ZarrSQLGraph(path)

    copied = graph.copy()

    assert isinstance(copied, ZarrSQLGraph)
    assert copied is not graph
    assert copied.node_ids() == node_ids
    with pytest.raises(TypeError, match="requires another ZarrSQLGraph"):
        ZarrSQLGraph.from_other(RustWorkXGraph())


def test_node_edge_and_compound_filters_use_xarray_sql(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    graph = ZarrSQLGraph(path)
    sql = MagicMock(wraps=graph._context.sql)
    graph._context.sql = sql

    assert graph.filter(NodeAttr("score") >= 1.0).node_ids() == node_ids[1:]
    assert graph.filter(NodeAttr("is_even") == True).node_ids() == [node_ids[0], node_ids[2]]
    assert graph.filter(EdgeAttr("weight") > 1).edge_ids() == [1]
    result = graph.filter(NodeAttr("score") > 0, EdgeAttr("weight") < 1).edge_attrs()
    assert result[DEFAULT_ATTR_KEYS.EDGE_ID].to_list() == [0]
    assert any("JOIN node" in call.args[0] for call in sql.call_args_list)


def test_neighbors_degrees_and_edge_lookup(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    graph = ZarrSQLGraph(path)

    assert graph.successors(node_ids[0]) == [node_ids[1]]
    assert graph.predecessors(node_ids[2]) == [node_ids[1]]
    successor_attrs = graph.successors(node_ids[0], attr_keys=["score"], return_attrs=True)
    assert successor_attrs.to_dicts() == [{"score": pytest.approx(1.5)}]
    assert graph.in_degree() == [0, 1, 1]
    assert graph.out_degree(node_ids[1]) == 1
    assert graph.has_node(node_ids[1])
    assert not graph.has_node(999)
    assert graph.has_edge(node_ids[0], node_ids[1])
    assert not graph.has_edge(node_ids[0], node_ids[2])
    assert graph.edge_id(node_ids[0], node_ids[1]) == 0
    assert graph.edge_list() == [[node_ids[0], node_ids[1]], [node_ids[1], node_ids[2]]]
    with pytest.raises(ValueError, match="does not exist"):
        graph.edge_id(node_ids[0], node_ids[2])


def test_metadata_key_maps_from_geff_and_subgraph(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    graph, metadata = ZarrSQLGraph.from_geff(
        path,
        node_attr_key_map={"score": "confidence"},
        edge_attr_key_map={"weight": "cost"},
    )

    assert metadata is graph._geff_metadata
    assert "confidence" in metadata.node_props_metadata
    assert "score" not in metadata.node_props_metadata
    assert "cost" in metadata.edge_props_metadata
    assert "weight" not in metadata.edge_props_metadata
    assert "confidence" in graph.node_attr_keys()
    assert "score" not in graph.node_attr_keys()
    assert graph.node_attrs(attr_keys=["confidence"])["confidence"].to_list() == pytest.approx([0.5, 1.5, 2.5])
    assert graph.edge_attrs(attr_keys=["cost"])["cost"].to_list() == pytest.approx([0.25, 1.25])
    assert graph.metadata["dataset"] == "demo"
    assert graph.metadata["shape"] == [3, 16, 16]
    assert graph.metadata["geff"]["directed"] is True
    assert graph.filter(NodeAttr("confidence") < 2).subgraph().node_ids() == node_ids[:2]

    with pytest.raises(ValueError, match="geff_read_kwargs"):
        ZarrSQLGraph.from_geff(path, geff_read_kwargs={"structure_validation": False})


def test_varlength_masks_load_only_after_row_selection(tmp_path: Path) -> None:
    path = tmp_path / "masks.geff"
    source = RustWorkXGraph()
    source.add_node_attr_key(AttrSchema(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4)))
    source.add_node_attr_key(AttrSchema(DEFAULT_ATTR_KEYS.MASK, pl.Object))
    first_mask = np.asarray([[True, False], [False, True]])
    second_mask = np.asarray([[False, True], [True, False]])
    node_ids = source.bulk_add_nodes(
        [
            {
                DEFAULT_ATTR_KEYS.T: 0,
                DEFAULT_ATTR_KEYS.BBOX: np.asarray([0, 0, 2, 2]),
                DEFAULT_ATTR_KEYS.MASK: Mask(first_mask, [0, 0, 2, 2]),
            },
            {
                DEFAULT_ATTR_KEYS.T: 1,
                DEFAULT_ATTR_KEYS.BBOX: np.asarray([2, 3, 4, 5]),
                DEFAULT_ATTR_KEYS.MASK: Mask(second_mask, [2, 3, 4, 5]),
            },
        ]
    )
    source.to_geff(path)

    graph = ZarrSQLGraph(path)
    assert graph._node_attr_schemas()[DEFAULT_ATTR_KEYS.MASK].dtype == pl.Object
    attrs = graph.filter(node_ids=[node_ids[1]]).node_attrs(
        attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.MASK]
    )

    assert attrs[DEFAULT_ATTR_KEYS.NODE_ID].to_list() == [node_ids[1]]
    loaded_mask = attrs[DEFAULT_ATTR_KEYS.MASK][0]
    assert isinstance(loaded_mask, Mask)
    np.testing.assert_array_equal(loaded_mask.mask, second_mask)
    np.testing.assert_array_equal(loaded_mask.bbox, [2, 3, 4, 5])


def test_mutations_and_metadata_writes_are_rejected(geff_graph: tuple[list[int], Path]) -> None:
    node_ids, path = geff_graph
    graph = ZarrSQLGraph(path)

    with pytest.raises(RuntimeError, match="read-only"):
        graph.add_node(
            {
                DEFAULT_ATTR_KEYS.T: 3,
                "score": 3.5,
                "label": 13,
                "is_even": False,
                DEFAULT_ATTR_KEYS.BBOX: [3, 4, 5, 6],
            }
        )
    with pytest.raises(RuntimeError, match="read-only"):
        graph.remove_edge(edge_id=graph.edge_ids()[0])
    with pytest.raises(RuntimeError, match="read-only"):
        graph.update_node_attrs(attrs={"score": 0.0}, node_ids=[node_ids[0]])
    with pytest.raises(RuntimeError, match="read-only"):
        graph.create_node_attr_index("score")
    with pytest.raises(RuntimeError, match="read-only"):
        graph.metadata["new"] = "value"
