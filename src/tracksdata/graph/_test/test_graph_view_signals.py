"""Tests for GraphView signal-emission consistency.

When a node-mutation signal fires through a GraphView, listeners attached to
either the root or the view must see the two graphs in a consistent state.
A listener attached to root that queries the view (or vice versa) must not
observe ghost or stale nodes.

See .claude/bugs/bugs_graphview.md (P1-1) for context.
"""

import polars as pl

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph


def test_view_node_signals_fire_with_consistent_state(graph_backend: BaseGraph) -> None:
    """add_node / remove_node: when either signal fires (on root or view), the
    two graphs must agree on `has_node`.

    Today this fails on `remove_node` because `GraphView.remove_node` does not
    block the root signal — root emits while the view's local rx_graph still
    holds the node.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    observations: list = []

    def make_slot(source: str, signal: str):
        def slot(node_id: int, *_args) -> None:
            observations.append((source, signal, node_id, graph_backend.has_node(node_id), view.has_node(node_id)))

        return slot

    graph_backend.node_added.connect(make_slot("root", "added"))
    graph_backend.node_removed.connect(make_slot("root", "removed"))
    view.node_added.connect(make_slot("view", "added"))
    view.node_removed.connect(make_slot("view", "removed"))

    new_id = view.add_node({"t": 1, "x": 1.0})
    view.remove_node(new_id)

    inconsistent = [obs for obs in observations if obs[3] != obs[4]]
    detail = "\n".join(
        f"  {source}.{signal}(node={nid}): root.has_node={rh}, view.has_node={vh}"
        for source, signal, nid, rh, vh in inconsistent
    )
    assert not inconsistent, f"Listener saw root and view in inconsistent state at signal time:\n{detail}"


def test_view_update_node_attrs_signal_fires_with_consistent_value(graph_backend: BaseGraph) -> None:
    """update_node_attrs: when either signal fires, root and view must hold
    the same value for the updated attribute.

    This used to fail on backends where root and view do not share an attribute
    storage (SQLGraph): root emits with the new value while the view's local
    rx_graph still holds the old one.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    observations: list = []

    def attr_value(graph: BaseGraph, nid: int) -> float:
        df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "x"])
        return df.filter(pl.col(DEFAULT_ATTR_KEYS.NODE_ID) == nid)["x"].item()

    def make_slot(source: str):
        def slot(nid: int, _old: dict, _new: dict) -> None:
            observations.append((source, nid, attr_value(graph_backend, nid), attr_value(view, nid)))

        return slot

    graph_backend.node_updated.connect(make_slot("root"))
    view.node_updated.connect(make_slot("view"))

    view.update_node_attrs(attrs={"x": 5.0}, node_ids=[node_id])

    inconsistent = [obs for obs in observations if obs[2] != obs[3]]
    detail = "\n".join(
        f"  {source}.node_updated(node={nid}): root.x={rx}, view.x={vx}" for source, nid, rx, vx in inconsistent
    )
    assert not inconsistent, f"Listener saw root and view holding different attribute values at signal time:\n{detail}"
