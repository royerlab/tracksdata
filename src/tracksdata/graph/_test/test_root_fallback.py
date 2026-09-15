"""Tests for `GraphView(root_fallback=True)`: reading node attribute keys a
partial view was built without, by falling back to its root graph."""

import numpy as np
import polars as pl
import pytest

from tracksdata.array import GraphArrayView
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph, GraphView
from tracksdata.nodes._mask import Mask

LEAN_KEYS = ["x", "y"]


def _populate(graph: BaseGraph) -> BaseGraph:
    """A graph with a cheap key set and one key a lean view will leave out."""
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_node_attr_key("label", dtype=pl.String, default_value="")
    graph.add_node_attr_key("solution", dtype=pl.Boolean, default_value=False)
    graph.add_node_attr_key("vec", dtype=pl.Array(pl.Int64, 2))

    for i in range(6):
        graph.add_node(
            {
                DEFAULT_ATTR_KEYS.T: i,
                "x": float(i),
                "y": float(i * 10),
                "label": f"n{i}",
                "solution": i % 2 == 0,
                "vec": np.array([i, -i]),
            }
        )
    return graph


def _views(graph: BaseGraph, **kwargs) -> tuple[GraphView, GraphView]:
    """A lean view and the equivalent full view, over the same nodes."""
    lean = graph.filter(NodeAttr("solution") == True).subgraph(node_attr_keys=LEAN_KEYS, root_fallback=True, **kwargs)
    full = graph.filter(NodeAttr("solution") == True).subgraph()
    return lean, full


# --- correctness of the fallback ------------------------------------------


def test_fallback_matches_full_view(graph_backend: BaseGraph) -> None:
    lean, full = _views(_populate(graph_backend))

    assert "label" not in lean.node_attr_keys()

    keys = [DEFAULT_ATTR_KEYS.NODE_ID, "label"]
    assert lean.node_attrs(attr_keys=keys).equals(full.node_attrs(attr_keys=keys))


def test_mixed_request_keeps_requested_order(graph_backend: BaseGraph) -> None:
    lean, full = _views(_populate(graph_backend))

    keys = ["label", "x", DEFAULT_ATTR_KEYS.NODE_ID, "y"]
    df = lean.node_attrs(attr_keys=keys)

    assert df.columns == keys
    assert df.equals(full.node_attrs(attr_keys=keys))


def test_fallback_without_node_id_requested(graph_backend: BaseGraph) -> None:
    """NODE_ID is the join key but must not leak into the result."""
    lean, full = _views(_populate(graph_backend))

    df = lean.node_attrs(attr_keys=["label", "x"])

    assert df.columns == ["label", "x"]
    assert df.equals(full.node_attrs(attr_keys=["label", "x"]))


def test_fallback_single_str_attr_key(graph_backend: BaseGraph) -> None:
    lean, full = _views(_populate(graph_backend))

    df = lean.node_attrs(attr_keys="label")

    assert df.columns == ["label"]
    assert df.equals(full.node_attrs(attr_keys="label"))


def test_fallback_unpack(graph_backend: BaseGraph) -> None:
    lean, full = _views(_populate(graph_backend))

    keys = ["x", "vec"]
    df = lean.node_attrs(attr_keys=keys, unpack=True)

    assert df.equals(full.node_attrs(attr_keys=keys, unpack=True))
    assert "vec" not in df.columns


def test_fallback_node_ids_subset(graph_backend: BaseGraph) -> None:
    lean, full = _views(_populate(graph_backend))
    subset = lean.node_ids()[:2]

    keys = [DEFAULT_ATTR_KEYS.NODE_ID, "label"]
    df = lean.filter(node_ids=subset).node_attrs(attr_keys=keys)

    assert df[DEFAULT_ATTR_KEYS.NODE_ID].to_list() == subset
    assert df.equals(full.filter(node_ids=subset).node_attrs(attr_keys=keys))


def test_fallback_never_returns_rows_outside_the_view(graph_backend: BaseGraph) -> None:
    """The root holds the `solution=False` nodes the view dropped."""
    graph = _populate(graph_backend)
    lean, _ = _views(graph)

    assert graph.num_nodes() > lean.num_nodes()

    df = lean.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "label"])

    assert df.height == lean.num_nodes()
    assert set(df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()) == set(lean.node_ids())


def test_fallback_row_order_matches_local_columns(graph_backend: BaseGraph) -> None:
    """Callers zip columns positionally; the join must not reorder rows."""
    lean, _ = _views(_populate(graph_backend))

    df = lean.node_attrs(attr_keys=["x", "label"])

    for x, label in zip(df["x"], df["label"], strict=True):
        assert label == f"n{int(x)}"


def test_fallback_after_local_update_is_not_stale(graph_backend: BaseGraph) -> None:
    lean, _ = _views(_populate(graph_backend))
    node_ids = lean.node_ids()

    lean.update_node_attrs(node_ids=node_ids, attrs={"x": 99.0})

    df = lean.node_attrs(attr_keys=["x", "label"])
    assert df["x"].to_list() == [99.0] * len(node_ids)
    assert df["label"].to_list() == sorted(df["label"].to_list())


def test_fallback_skips_node_removed_from_view(graph_backend: BaseGraph) -> None:
    graph = _populate(graph_backend)
    lean, _ = _views(graph)

    removed = lean.node_ids()[0]
    lean.remove_node_from_view(removed)

    df = lean.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "label"])

    assert removed not in df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    assert graph.has_node(removed)


def test_attr_keys_none_stays_lean(graph_backend: BaseGraph) -> None:
    """`None` means "what this view holds", not "everything the root has"."""
    lean, _ = _views(_populate(graph_backend))

    assert "label" not in lean.node_attrs().columns


# --- error behaviour preserved --------------------------------------------


def test_unknown_key_still_raises(graph_backend: BaseGraph) -> None:
    lean, _ = _views(_populate(graph_backend))

    with pytest.raises(KeyError, match="not found"):
        lean.node_attrs(attr_keys=["nowhere"])


def test_without_fallback_missing_key_raises_with_hint(graph_backend: BaseGraph) -> None:
    graph = _populate(graph_backend)
    view = graph.filter(NodeAttr("solution") == True).subgraph(node_attr_keys=LEAN_KEYS)

    with pytest.raises(KeyError, match="root_fallback=True"):
        view.node_attrs(attr_keys=["label"])


def test_without_fallback_unknown_key_has_no_hint(graph_backend: BaseGraph) -> None:
    graph = _populate(graph_backend)
    view = graph.filter(NodeAttr("solution") == True).subgraph(node_attr_keys=LEAN_KEYS)

    with pytest.raises(KeyError) as excinfo:
        view.node_attrs(attr_keys=["nowhere"])

    assert "root_fallback" not in str(excinfo.value)


def test_update_on_missing_key_writes_to_root_and_reads_back(graph_backend: BaseGraph) -> None:
    """A write to a non-local key already goes to the root (pre-existing
    behaviour: `_update_local_node_attrs` skips keys the view does not hold).
    Fallback cannot make that stale -- there is no local copy to disagree."""
    graph = _populate(graph_backend)
    lean, _ = _views(graph)
    node_ids = lean.node_ids()

    lean.update_node_attrs(node_ids=node_ids, attrs={"label": "written"})

    assert lean.node_attrs(attr_keys=["label"])["label"].to_list() == ["written"] * len(node_ids)
    assert graph.filter(node_ids=node_ids).node_attrs(attr_keys=["label"])["label"].to_list() == ["written"] * len(
        node_ids
    )


def test_node_attr_keys_reports_only_local_keys(graph_backend: BaseGraph) -> None:
    lean, _ = _views(_populate(graph_backend))

    assert sorted(lean.node_attr_keys()) == sorted([*LEAN_KEYS, DEFAULT_ATTR_KEYS.T])


def test_fallback_is_off_by_default(graph_backend: BaseGraph) -> None:
    view = _populate(graph_backend).filter(NodeAttr("solution") == True).subgraph(node_attr_keys=LEAN_KEYS)

    assert view._root_fallback is False


# --- end to end through GraphArrayView -------------------------------------


def test_graph_array_view_on_lean_view(graph_backend: BaseGraph) -> None:
    """The case the feature exists for: render masks the view does not hold."""
    graph = graph_backend
    graph.add_node_attr_key("label", dtype=pl.Int64)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))
    graph.add_node_attr_key("solution", dtype=pl.Boolean, default_value=False)

    for i, solution in enumerate([True, False, True]):
        mask = Mask(
            np.ones((2, 2), dtype=bool),
            bbox=np.array([10 * i, 20, 10 * i + 2, 22]),
        )
        graph.add_node(
            {
                DEFAULT_ATTR_KEYS.T: 0,
                "label": i + 1,
                DEFAULT_ATTR_KEYS.MASK: mask,
                DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
                "solution": solution,
            }
        )

    node_filter = graph.filter(NodeAttr("solution") == True)
    # bbox stays local: the view's own R-tree is built from it.
    lean = node_filter.subgraph(node_attr_keys=["label", DEFAULT_ATTR_KEYS.BBOX], root_fallback=True)
    full = node_filter.subgraph()

    shape = (1, 100, 100)
    lean_array = np.asarray(GraphArrayView(graph=lean, shape=shape, attr_key="label")[0])
    full_array = np.asarray(GraphArrayView(graph=full, shape=shape, attr_key="label")[0])

    assert lean_array.max() > 0
    np.testing.assert_array_equal(lean_array, full_array)


# --- regression guard: the excluded column is never materialized ------------


class _ExplodingBlob:
    """Unpickles only if something actually reads the column it lives in."""

    def __setstate__(self, state: dict) -> None:
        raise AssertionError("excluded blob column was read while building the lean view")

    def __getstate__(self) -> dict:
        return {}


def test_lean_view_does_not_read_the_excluded_column() -> None:
    from tracksdata.graph import SQLGraph

    graph = SQLGraph(drivername="sqlite", database=":memory:")
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("blob", dtype=pl.Object)

    for i in range(4):
        graph.add_node({DEFAULT_ATTR_KEYS.T: i, "x": float(i), "blob": _ExplodingBlob()})

    # Would raise if `blob` were fetched and unpickled here.
    lean = graph.filter().subgraph(node_attr_keys=["x"], root_fallback=True)

    assert "blob" not in lean.node_attr_keys()
    # ... and it is genuinely absent from the view's own storage, not just hidden.
    assert all("blob" not in payload for payload in lean.rx_graph.nodes())

    with pytest.raises(AssertionError, match="excluded blob column"):
        lean.node_attrs(attr_keys=["blob"])
