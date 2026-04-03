"""Tests for shift_division in tracksdata.functional."""

import numpy as np
import polars as pl
import pytest

import tracksdata as td
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional import shift_division
from tracksdata.nodes._mask import Mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_graph() -> tuple[td.graph.RustWorkXGraph, dict[str, int]]:
    """
    Build a minimal division graph:

        gp (t=0, x=0.0)
         |
        p  (t=1, x=2.0)
         |
        d  (t=2, x=4.0)   ← dividing node
       / \\
      c1   c2              (t=3, x=2.0 / x=6.0)

    Returns the graph and a dict of node IDs keyed by name.
    """
    g = td.graph.RustWorkXGraph()
    g.add_node_attr_key("x", pl.Float64)

    ids: dict[str, int] = {}
    ids["gp"] = g.add_node({"t": 0, "x": 0.0})
    ids["p"] = g.add_node({"t": 1, "x": 2.0})
    ids["d"] = g.add_node({"t": 2, "x": 4.0})
    ids["c1"] = g.add_node({"t": 3, "x": 2.0})
    ids["c2"] = g.add_node({"t": 3, "x": 6.0})

    g.add_edge(ids["gp"], ids["p"], {})
    g.add_edge(ids["p"], ids["d"], {})
    g.add_edge(ids["d"], ids["c1"], {})
    g.add_edge(ids["d"], ids["c2"], {})

    return g, ids


def _make_deep_graph() -> tuple[td.graph.RustWorkXGraph, dict[str, int]]:
    """
    Extends the simple graph with grandchildren for multi-frame tests:

        gp (t=0, x=0.0) → p (t=1, x=2.0) → d (t=2, x=4.0)
                                                   / \\
                                       c1 (t=3, x=2.0)  c2 (t=3, x=6.0)
                                        |                  |
                                   g1 (t=4, x=1.0)   g2 (t=4, x=7.0)
    """
    g, ids = _make_simple_graph()
    ids["g1"] = g.add_node({"t": 4, "x": 1.0})
    ids["g2"] = g.add_node({"t": 4, "x": 7.0})
    g.add_edge(ids["c1"], ids["g1"], {})
    g.add_edge(ids["c2"], ids["g2"], {})
    return g, ids


def _node_attrs(graph: td.graph.RustWorkXGraph, node_id: int) -> dict:
    return graph.nodes[node_id].to_dict()


# ---------------------------------------------------------------------------
# Tests: frames=0
# ---------------------------------------------------------------------------


def test_shift_division_zero_frames_returns_copy() -> None:
    g, ids = _make_simple_graph()
    result = shift_division(g, ids["d"], frames=0)

    # Must be a different object
    assert result is not g

    # Structure must be identical
    assert set(result.node_ids()) == set(g.node_ids())
    assert result.successors(ids["d"]) == g.successors(ids["d"])


def test_shift_division_does_not_modify_original() -> None:
    g, ids = _make_simple_graph()
    original_node_ids = set(g.node_ids())

    shift_division(g, ids["d"], frames=1)

    assert set(g.node_ids()) == original_node_ids


# ---------------------------------------------------------------------------
# Tests: ahead (positive frames)
# ---------------------------------------------------------------------------


def test_shift_division_ahead_1_frame() -> None:
    g, ids = _make_deep_graph()
    result = shift_division(g, ids["d"], frames=1)

    # c1 and c2 are removed, one merged node is added → net -1
    assert len(result.node_ids()) == len(g.node_ids()) - 1

    # d still exists, its single successor is the merged node
    successors_d = result.successors(ids["d"])
    assert len(successors_d) == 1
    merged_id = successors_d[0]

    # Merged node attrs: average of c1 and c2
    merged = _node_attrs(result, merged_id)
    assert merged["t"] == 3  # avg(3, 3) = 3
    assert merged["x"] == pytest.approx(4.0)  # avg(2.0, 6.0)

    # Merged node inherits g1 and g2 as successors
    assert set(result.successors(merged_id)) == {ids["g1"], ids["g2"]}

    # c1 and c2 are gone
    assert ids["c1"] not in result.node_ids()
    assert ids["c2"] not in result.node_ids()


def test_shift_division_ahead_2_frames() -> None:
    g, ids = _make_deep_graph()
    result = shift_division(g, ids["d"], frames=2)

    # After 2 steps: d → merged_c → merged_g (no more successors)
    successors_d = result.successors(ids["d"])
    assert len(successors_d) == 1
    merged_c = successors_d[0]

    successors_merged_c = result.successors(merged_c)
    assert len(successors_merged_c) == 1
    merged_g = successors_merged_c[0]

    # merged_g is average of g1 and g2
    attrs = _node_attrs(result, merged_g)
    assert attrs["t"] == 4  # avg(4, 4)
    assert attrs["x"] == pytest.approx(4.0)  # avg(1.0, 7.0)

    # No further successors
    assert result.successors(merged_g) == []

    # 7 original nodes - 4 removed (c1,c2,g1,g2) + 2 added (merged_c, merged_g) = 5
    assert len(result.node_ids()) == len(g.node_ids()) - 2


# ---------------------------------------------------------------------------
# Tests: behind (negative frames)
# ---------------------------------------------------------------------------


def test_shift_division_behind_1_frame() -> None:
    g, ids = _make_simple_graph()
    result = shift_division(g, ids["d"], frames=-1)

    # d is removed, two replacement nodes are added → net +1
    assert len(result.node_ids()) == len(g.node_ids()) + 1
    assert ids["d"] not in result.node_ids()

    # p now has two children at t=2
    new_children = result.successors(ids["p"])
    assert len(new_children) == 2

    attrs_list = [_node_attrs(result, nid) for nid in new_children]
    x_values = sorted(a["x"] for a in attrs_list)
    t_values = [a["t"] for a in attrs_list]

    # Both replacement nodes are at the original divider's time
    assert all(t == 2 for t in t_values)

    # x = interp(p.x=2.0, c1.x=2.0) = 2.0 and interp(p.x=2.0, c2.x=6.0) = 4.0
    assert x_values == pytest.approx([2.0, 4.0])

    # Each replacement node connects to exactly one of the original children
    successors_of_new = {result.successors(nid)[0] for nid in new_children}
    assert successors_of_new == {ids["c1"], ids["c2"]}


def test_shift_division_behind_2_frames() -> None:
    g, ids = _make_simple_graph()
    result = shift_division(g, ids["d"], frames=-2)

    # 5 original - 2 removed (d, p) + 4 added (d1',d2',p1',p2') = 7
    assert len(result.node_ids()) == len(g.node_ids()) + 2

    # gp has two children (the p-replacements)
    gp_children = result.successors(ids["gp"])
    assert len(gp_children) == 2

    # Each p-replacement has one child (a d-replacement)
    d_replacements = set()
    for pc in gp_children:
        children = result.successors(pc)
        assert len(children) == 1
        d_replacements.add(children[0])
    assert len(d_replacements) == 2

    # Each d-replacement connects to one of the original children (c1 or c2)
    leaf_targets = {result.successors(dr)[0] for dr in d_replacements}
    assert leaf_targets == {ids["c1"], ids["c2"]}

    # p-replacements are at t=1: interp(gp.t=0, d-replacement.t=2) = 1
    p_attrs = [_node_attrs(result, pc) for pc in gp_children]
    assert all(a["t"] == 1 for a in p_attrs)


# ---------------------------------------------------------------------------
# Tests: edge attributes preserved
# ---------------------------------------------------------------------------


def test_shift_division_ahead_preserves_edge_attrs() -> None:
    g = td.graph.RustWorkXGraph()
    g.add_node_attr_key("x", pl.Float64)
    g.add_edge_attr_key("weight", pl.Float64, default_value=0.0)

    p = g.add_node({"t": 0, "x": 0.0})
    d = g.add_node({"t": 1, "x": 1.0})
    c1 = g.add_node({"t": 2, "x": 0.0})
    c2 = g.add_node({"t": 2, "x": 2.0})
    g1 = g.add_node({"t": 3, "x": 0.0})
    g2 = g.add_node({"t": 3, "x": 2.0})

    g.add_edge(p, d, {"weight": 0.9})
    g.add_edge(d, c1, {"weight": 0.6})
    g.add_edge(d, c2, {"weight": 0.4})
    g.add_edge(c1, g1, {"weight": 0.8})
    g.add_edge(c2, g2, {"weight": 0.7})

    result = shift_division(g, d, frames=1)

    merged_id = result.successors(d)[0]

    # Edge d → merged: avg(0.6, 0.4) = 0.5
    eid = result.edge_id(d, merged_id)
    assert result.edges[eid]["weight"] == pytest.approx(0.5)

    # Edges merged → g1 and merged → g2 preserve original weights
    eid_g1 = result.edge_id(merged_id, g1)
    eid_g2 = result.edge_id(merged_id, g2)
    assert result.edges[eid_g1]["weight"] == pytest.approx(0.8)
    assert result.edges[eid_g2]["weight"] == pytest.approx(0.7)


def test_shift_division_behind_preserves_edge_attrs() -> None:
    g = td.graph.RustWorkXGraph()
    g.add_node_attr_key("x", pl.Float64)
    g.add_edge_attr_key("weight", pl.Float64, default_value=0.0)

    p = g.add_node({"t": 0, "x": 0.0})
    d = g.add_node({"t": 1, "x": 2.0})
    c1 = g.add_node({"t": 2, "x": 1.0})
    c2 = g.add_node({"t": 2, "x": 3.0})

    g.add_edge(p, d, {"weight": 0.9})
    g.add_edge(d, c1, {"weight": 0.6})
    g.add_edge(d, c2, {"weight": 0.4})

    result = shift_division(g, d, frames=-1)

    new_children = result.successors(p)
    assert len(new_children) == 2

    # Both p → replacement edges use the original p→d weight
    for nc in new_children:
        eid = result.edge_id(p, nc)
        assert result.edges[eid]["weight"] == pytest.approx(0.9)

    # Replacement → child edges preserve the original d→c weights
    for nc in new_children:
        child = result.successors(nc)[0]
        eid = result.edge_id(nc, child)
        expected = 0.6 if child == c1 else 0.4
        assert result.edges[eid]["weight"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


def test_shift_division_ahead_at_penultimate_frame() -> None:
    """Shifting ahead when children are leaf nodes (last frame, no grandchildren)."""
    g = td.graph.RustWorkXGraph()
    g.add_node_attr_key("x", pl.Float64)

    p = g.add_node({"t": 0, "x": 0.0})
    d = g.add_node({"t": 1, "x": 2.0})
    c1 = g.add_node({"t": 2, "x": 1.0})
    c2 = g.add_node({"t": 2, "x": 3.0})
    g.add_edge(p, d, {})
    g.add_edge(d, c1, {})
    g.add_edge(d, c2, {})

    result = shift_division(g, d, frames=1)

    # c1 and c2 merged into one leaf node
    successors_d = result.successors(d)
    assert len(successors_d) == 1
    merged_id = successors_d[0]

    assert result.successors(merged_id) == []
    assert result.nodes[merged_id]["x"] == pytest.approx(2.0)  # avg(1.0, 3.0)
    assert result.nodes[merged_id]["t"] == 2  # avg(2, 2)


def test_shift_division_ahead_raises_if_not_two_children() -> None:
    g = td.graph.RustWorkXGraph()
    n0 = g.add_node({"t": 0})
    n1 = g.add_node({"t": 1})
    g.add_edge(n0, n1, {})

    with pytest.raises(ValueError, match="exactly 2 children"):
        shift_division(g, n0, frames=1)


def test_shift_division_behind_raises_if_no_parent() -> None:
    """A dividing node at t=0 has no parent — shifting behind must raise."""
    g = td.graph.RustWorkXGraph()
    d = g.add_node({"t": 0})
    c1 = g.add_node({"t": 1})
    c2 = g.add_node({"t": 1})
    g.add_edge(d, c1, {})
    g.add_edge(d, c2, {})

    with pytest.raises(ValueError, match="exactly 1 parent"):
        shift_division(g, d, frames=-1)


def test_shift_division_behind_raises_if_not_two_children() -> None:
    g = td.graph.RustWorkXGraph()
    p = g.add_node({"t": 0})
    d = g.add_node({"t": 1})
    c1 = g.add_node({"t": 2})
    g.add_edge(p, d, {})
    g.add_edge(d, c1, {})

    with pytest.raises(ValueError, match="exactly 2 children"):
        shift_division(g, d, frames=-1)


# ---------------------------------------------------------------------------
# Tests: bbox and mask attributes
# ---------------------------------------------------------------------------


def _make_graph_with_bbox() -> tuple[td.graph.RustWorkXGraph, dict[str, int]]:
    """
    Graph with bbox (Array) attributes:

        p  (t=0, bbox=[0,0,4,4])
        |
        d  (t=1, bbox=[0,0,8,8])   ← dividing node
       / \\
      c1   c2   (t=2, bbox=[0,0,4,4] / [4,4,8,8])
    """
    g = td.graph.RustWorkXGraph()
    g.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, pl.Array(pl.Int64, 4))

    ids: dict[str, int] = {}
    ids["p"] = g.add_node({"t": 0, DEFAULT_ATTR_KEYS.BBOX: np.array([0, 0, 4, 4])})
    ids["d"] = g.add_node({"t": 1, DEFAULT_ATTR_KEYS.BBOX: np.array([0, 0, 8, 8])})
    ids["c1"] = g.add_node({"t": 2, DEFAULT_ATTR_KEYS.BBOX: np.array([0, 0, 4, 4])})
    ids["c2"] = g.add_node({"t": 2, DEFAULT_ATTR_KEYS.BBOX: np.array([4, 4, 8, 8])})
    g.add_edge(ids["p"], ids["d"], {})
    g.add_edge(ids["d"], ids["c1"], {})
    g.add_edge(ids["d"], ids["c2"], {})
    return g, ids


def _make_graph_with_mask() -> tuple[td.graph.RustWorkXGraph, dict[str, int]]:
    """
    Graph with Mask (Object) attributes.

        p  (t=0)
        |
        d  (t=1)   ← dividing node
       / \\
      c1   c2   (t=2)
    """
    g = td.graph.RustWorkXGraph()
    g.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, pl.Object)

    mask_p = Mask(np.ones((4, 4), dtype=bool), bbox=np.array([0, 0, 4, 4]))
    mask_d = Mask(np.ones((4, 4), dtype=bool), bbox=np.array([0, 0, 4, 4]))
    mask_c1 = Mask(np.ones((2, 2), dtype=bool), bbox=np.array([0, 0, 2, 2]))
    mask_c2 = Mask(np.ones((2, 2), dtype=bool), bbox=np.array([2, 2, 4, 4]))

    ids: dict[str, int] = {}
    ids["p"] = g.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask_p})
    ids["d"] = g.add_node({"t": 1, DEFAULT_ATTR_KEYS.MASK: mask_d})
    ids["c1"] = g.add_node({"t": 2, DEFAULT_ATTR_KEYS.MASK: mask_c1})
    ids["c2"] = g.add_node({"t": 2, DEFAULT_ATTR_KEYS.MASK: mask_c2})
    g.add_edge(ids["p"], ids["d"], {})
    g.add_edge(ids["d"], ids["c1"], {})
    g.add_edge(ids["d"], ids["c2"], {})
    return g, ids, {"p": mask_p, "d": mask_d, "c1": mask_c1, "c2": mask_c2}


def test_shift_division_ahead_bbox() -> None:
    g, ids = _make_graph_with_bbox()
    result = shift_division(g, ids["d"], frames=1)

    merged_id = result.successors(ids["d"])[0]
    merged_bbox = np.asarray(result.nodes[merged_id][DEFAULT_ATTR_KEYS.BBOX])

    # bbox is pl.Array; merged node's bbox is element-wise average of children
    # c1=[0,0,4,4], c2=[4,4,8,8] → avg=[2,2,6,6]
    np.testing.assert_array_equal(merged_bbox, np.array([2, 2, 6, 6]))

    # t is Int32 (numeric); merged node's t is average of children: avg(2, 2) = 2
    assert result.nodes[merged_id][DEFAULT_ATTR_KEYS.T] == 2


def test_shift_division_behind_bbox_is_interpolated() -> None:
    """
    bbox is a pl.Array column; each replacement node's bbox is element-wise
    average of the parent's bbox and its respective child's bbox.
    """
    g, ids = _make_graph_with_bbox()
    # p=[0,0,4,4], c1=[0,0,4,4], c2=[4,4,8,8]
    result = shift_division(g, ids["d"], frames=-1)

    new_children = result.successors(ids["p"])
    assert len(new_children) == 2

    got = {tuple(np.asarray(result.nodes[nc][DEFAULT_ATTR_KEYS.BBOX]).tolist()) for nc in new_children}
    # interp(p=[0,0,4,4], c1=[0,0,4,4]) = [0,0,4,4]
    # interp(p=[0,0,4,4], c2=[4,4,8,8]) = [2,2,6,6]
    assert got == {(0, 0, 4, 4), (2, 2, 6, 6)}


def test_shift_division_ahead_mask() -> None:
    g, ids, masks = _make_graph_with_mask()
    result = shift_division(g, ids["d"], frames=1)

    merged_id = result.successors(ids["d"])[0]
    merged_mask = result.nodes[merged_id][DEFAULT_ATTR_KEYS.MASK]
    assert isinstance(merged_mask, Mask)

    # Non-numeric fallback → first child's mask
    first_child = g.successors(ids["d"])[0]
    expected_mask = masks["c1"] if first_child == ids["c1"] else masks["c2"]
    assert merged_mask == expected_mask

    # p and d are untouched
    assert result.nodes[ids["p"]][DEFAULT_ATTR_KEYS.MASK] == masks["p"]
    assert result.nodes[ids["d"]][DEFAULT_ATTR_KEYS.MASK] == masks["d"]


def test_shift_division_behind_mask() -> None:
    g, ids, masks = _make_graph_with_mask()
    result = shift_division(g, ids["d"], frames=-1)

    new_children = result.successors(ids["p"])
    assert len(new_children) == 2

    # Each replacement node carries its respective child's mask
    got_masks = [result.nodes[nc][DEFAULT_ATTR_KEYS.MASK] for nc in new_children]
    expected_masks = [masks["c1"], masks["c2"]]
    assert all(any(g == e for e in expected_masks) for g in got_masks)
    assert all(any(e == g for g in got_masks) for e in expected_masks)

    # c1 and c2 are untouched
    assert result.nodes[ids["c1"]][DEFAULT_ATTR_KEYS.MASK] == masks["c1"]
    assert result.nodes[ids["c2"]][DEFAULT_ATTR_KEYS.MASK] == masks["c2"]
