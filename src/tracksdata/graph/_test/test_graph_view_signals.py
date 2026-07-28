"""Tests for GraphView signal-emission consistency.

When a node-mutation signal fires through a GraphView, listeners attached to
either the root or the view must see the two graphs in a consistent state.
A listener attached to root that queries the view (or vice versa) must not
observe ghost or stale nodes.

The second half of this module covers the *reverse* direction: a GraphView is
meant to be a live view, so writes made directly to the **root** must be
visible in the view and must re-emit ``view.node_updated`` for the nodes the
view contains. Edge attributes propagate the same way, minus the notification —
there is no ``edge_updated`` signal.
"""

import gc
import pickle

import polars as pl
import pytest

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph, RustWorkXGraph
from tracksdata.graph._rustworkx_graph import IndexedRXGraph


def test_node_signals_fire_after_the_emitting_graph_is_updated(graph_backend: BaseGraph) -> None:
    """add_node / remove_node: a signal must reflect the graph that emitted it.

    Each graph is responsible for its own signal only. A listener on the root
    sees the root updated; a listener on the view sees the view updated. Neither
    is required to observe the *other* graph in any particular state — see
    royerlab/tracksdata#324.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    observations: list = []

    def make_slot(graph: BaseGraph, source: str, signal: str):
        def slot(node_ids: list[int], *_args) -> None:
            for node_id in node_ids:
                observations.append((source, signal, node_id, graph.has_node(node_id)))

        return slot

    graph_backend.node_added.connect(make_slot(graph_backend, "root", "added"))
    graph_backend.node_removed.connect(make_slot(graph_backend, "root", "removed"))
    view.node_added.connect(make_slot(view, "view", "added"))
    view.node_removed.connect(make_slot(view, "view", "removed"))

    new_id = view.add_node({"t": 1, "x": 1.0})
    view.remove_node(new_id)

    # every "added" must see the node present, every "removed" must see it gone,
    # each in the graph that emitted the signal
    wrong = [obs for obs in observations if obs[3] != (obs[1] == "added")]
    detail = "\n".join(
        f"  {source}.{signal}(node={nid}): {source}.has_node={present}" for source, signal, nid, present in wrong
    )
    assert not wrong, f"Signal did not reflect the state of the graph that emitted it:\n{detail}"
    # both graphs emitted both events
    assert {(source, signal) for source, signal, _, _ in observations} == {
        ("root", "added"),
        ("root", "removed"),
        ("view", "added"),
        ("view", "removed"),
    }


def test_update_node_attrs_signal_reflects_the_emitting_graph(graph_backend: BaseGraph) -> None:
    """update_node_attrs: each graph's signal must carry that graph's new value.

    As with add/remove, a listener is only promised that the graph it subscribed
    to is current — not that root and view agree at that instant.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    observations: list = []

    def attr_value(graph: BaseGraph, nid: int) -> float:
        df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, "x"])
        return df.filter(pl.col(DEFAULT_ATTR_KEYS.NODE_ID) == nid)["x"].item()

    def make_slot(graph: BaseGraph, source: str):
        def slot(node_ids: list[int], _old: list[dict], _new: list[dict], *_rest) -> None:
            for nid in node_ids:
                observations.append((source, nid, attr_value(graph, nid)))

        return slot

    graph_backend.node_updated.connect(make_slot(graph_backend, "root"))
    view.node_updated.connect(make_slot(view, "view"))

    view.update_node_attrs(attrs={"x": 5.0}, node_ids=[node_id])

    stale = [obs for obs in observations if obs[2] != 5.0]
    detail = "\n".join(f"  {source}.node_updated(node={nid}): {source}.x={value}" for source, nid, value in stale)
    assert not stale, f"Signal fired before the emitting graph held the new value:\n{detail}"
    assert {source for source, _, _ in observations} == {"root", "view"}


# --------------------------------------------------------------------------
# Root -> view propagation ("the view is a live view")
# --------------------------------------------------------------------------


class _UpdateRecorder:
    """Collects every ``node_updated`` emission from a graph.

    Holds a reference to its own bound slot so the connection is not dropped
    by signal implementations that keep only weak references to listeners.
    """

    def __init__(self, graph: BaseGraph) -> None:
        self.calls: list[tuple[list[int], list[dict], list[dict], set[str]]] = []
        graph.node_updated.connect(self._slot)

    def _slot(self, node_ids: list[int], old: list[dict], new: list[dict], changed_keys: set[str]) -> None:
        self.calls.append((list(node_ids), old, new, changed_keys))

    def __len__(self) -> int:
        return len(self.calls)


def _record_updates(graph: BaseGraph) -> _UpdateRecorder:
    """Connect to ``graph.node_updated`` and collect every emission."""
    return _UpdateRecorder(graph)


def _value_of(graph: BaseGraph, node_id: int, key: str = "x") -> float:
    df = graph.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, key])
    return df.filter(pl.col(DEFAULT_ATTR_KEYS.NODE_ID) == node_id)[key].item()


def test_root_update_is_visible_in_view(graph_backend: BaseGraph) -> None:
    """A write directly to the root must be readable through the view."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()

    graph_backend.update_node_attrs(attrs={"x": 7.0}, node_ids=[node_id])

    assert _value_of(graph_backend, node_id) == 7.0
    assert _value_of(view, node_id) == 7.0, "view returned a stale value after a root write"


def test_root_update_emits_view_signal(graph_backend: BaseGraph) -> None:
    """A write to the root must re-emit ``node_updated`` on the view."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    view_calls = _record_updates(view)

    graph_backend.update_node_attrs(attrs={"x": 3.5}, node_ids=[node_id])

    assert len(view_calls) == 1, f"expected exactly one view emission, got {len(view_calls)}"
    node_ids, old, new, changed_keys = view_calls.calls[0]
    assert node_ids == [node_id]
    assert old[0]["x"] == 0.0
    assert new[0]["x"] == 3.5
    assert changed_keys == {"x"}


def test_root_update_reports_view_node_ids(graph_backend: BaseGraph) -> None:
    """Forwarded events must use the IDs the view exposes, not internal ones.

    Relevant for ``IndexedRXGraph`` roots, where the graph's external node IDs
    differ from the underlying rustworkx indices.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_ids = [graph_backend.add_node({"t": t, "x": 0.0}) for t in range(3)]

    view = graph_backend.filter().subgraph()
    view_calls = _record_updates(view)

    graph_backend.update_node_attrs(attrs={"x": 1.0}, node_ids=[node_ids[1]])

    assert len(view_calls) == 1
    reported = view_calls.calls[0][0]
    assert reported == [node_ids[1]]
    # the reported id must be addressable on the view
    assert view.has_node(reported[0])
    assert _value_of(view, reported[0]) == 1.0


def test_root_update_outside_view_does_not_emit(graph_backend: BaseGraph) -> None:
    """Updating a node absent from the view must not emit on the view."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    inside = graph_backend.add_node({"t": 0, "x": 0.0})
    outside = graph_backend.add_node({"t": 5, "x": 0.0})

    view = graph_backend.filter(node_ids=[inside]).subgraph()
    assert not view.has_node(outside)

    view_calls = _record_updates(view)

    graph_backend.update_node_attrs(attrs={"x": 9.0}, node_ids=[outside])

    assert view_calls.calls == [], "view emitted for a node it does not contain"


def test_root_update_mixed_batch_filters_to_view_nodes(graph_backend: BaseGraph) -> None:
    """A batch spanning in- and out-of-view nodes emits only the in-view subset."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    inside = graph_backend.add_node({"t": 0, "x": 0.0})
    outside = graph_backend.add_node({"t": 5, "x": 0.0})

    view = graph_backend.filter(node_ids=[inside]).subgraph()
    view_calls = _record_updates(view)

    graph_backend.update_node_attrs(attrs={"x": 4.0}, node_ids=[inside, outside])

    assert len(view_calls) == 1
    node_ids, _old, new, _keys = view_calls.calls[0]
    assert node_ids == [inside]
    assert new[0]["x"] == 4.0
    # both nodes were still written on the root
    assert _value_of(graph_backend, outside) == 4.0


def test_root_update_sequence_values_map_per_node(graph_backend: BaseGraph) -> None:
    """Per-node sequence values must be reported against the matching node."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    first = graph_backend.add_node({"t": 0, "x": 0.0})
    second = graph_backend.add_node({"t": 1, "x": 0.0})

    view = graph_backend.filter().subgraph()
    view_calls = _record_updates(view)

    graph_backend.update_node_attrs(attrs={"x": [11.0, 22.0]}, node_ids=[first, second])

    assert len(view_calls) == 1
    node_ids, _old, new, _keys = view_calls.calls[0]
    by_id = dict(zip(node_ids, [n["x"] for n in new], strict=True))
    assert by_id == {first: 11.0, second: 22.0}
    assert _value_of(view, first) == 11.0
    assert _value_of(view, second) == 22.0


def test_root_update_notifies_multiple_sibling_views(graph_backend: BaseGraph) -> None:
    """Every registered view containing the node must be notified."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    shared = graph_backend.add_node({"t": 0, "x": 0.0})
    other = graph_backend.add_node({"t": 5, "x": 0.0})

    view_a = graph_backend.filter().subgraph()
    view_b = graph_backend.filter(node_ids=[shared]).subgraph()
    view_c = graph_backend.filter(node_ids=[other]).subgraph()

    calls_a = _record_updates(view_a)
    calls_b = _record_updates(view_b)
    calls_c = _record_updates(view_c)

    graph_backend.update_node_attrs(attrs={"x": 6.0}, node_ids=[shared])

    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert calls_c.calls == [], "view without the node should not be notified"


def test_view_write_emits_exactly_once_on_each_graph(graph_backend: BaseGraph) -> None:
    """Writing through the view must not double-emit via the root registry.

    The view unregisters itself from the root while delegating the write, so
    the event is emitted once by the view itself rather than twice.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    root_calls = _record_updates(graph_backend)
    view_calls = _record_updates(view)

    view.update_node_attrs(attrs={"x": 2.0}, node_ids=[node_id])

    assert len(root_calls) == 1, f"root emitted {len(root_calls)} times, expected 1"
    assert len(view_calls) == 1, f"view emitted {len(view_calls)} times, expected 1"
    assert view_calls.calls[0][2][0]["x"] == 2.0


def test_view_is_reregistered_after_writing_through_it(graph_backend: BaseGraph) -> None:
    """A write through the view must not permanently unregister it."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()

    # write through the view first, which temporarily unregisters it
    view.update_node_attrs(attrs={"x": 1.0}, node_ids=[node_id])

    view_calls = _record_updates(view)
    # a subsequent *root* write must still reach the view
    graph_backend.update_node_attrs(attrs={"x": 8.0}, node_ids=[node_id])

    assert len(view_calls) == 1, "view stopped receiving root updates after writing through it"
    assert view_calls.calls[0][2][0]["x"] == 8.0


def test_root_update_with_no_view_listener_is_harmless(graph_backend: BaseGraph) -> None:
    """Forwarding must be a no-op when nothing is connected to the view."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()

    graph_backend.update_node_attrs(attrs={"x": 5.0}, node_ids=[node_id])

    assert _value_of(view, node_id) == 5.0


def test_nested_view_receives_root_updates(graph_backend: BaseGraph) -> None:
    """A subgraph taken from a view must also observe root writes."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    outer = graph_backend.filter().subgraph()
    inner = outer.filter().subgraph()

    outer_calls = _record_updates(outer)
    inner_calls = _record_updates(inner)

    graph_backend.update_node_attrs(attrs={"x": 4.5}, node_ids=[node_id])

    assert len(outer_calls) == 1
    assert len(inner_calls) == 1, "nested view did not receive the root update"
    assert _value_of(inner, node_id) == 4.5


def test_unreferenced_view_is_released(graph_backend: BaseGraph) -> None:
    """Dropping the last reference to a view must unregister it.

    The registry holds views weakly, so an unreachable view stops costing
    anything on subsequent root updates.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    assert len(graph_backend._views) == 1

    del view
    gc.collect()

    assert len(graph_backend._views) == 0, "view was not released from the registry"

    # updating with a released view must not raise
    graph_backend.update_node_attrs(attrs={"x": 1.0}, node_ids=[node_id])
    assert _value_of(graph_backend, node_id) == 1.0


def test_referenced_view_is_kept_registered(graph_backend: BaseGraph) -> None:
    """A view that is still referenced must survive garbage collection."""
    graph_backend.add_node_attr_key("x", pl.Float64)
    node_id = graph_backend.add_node({"t": 0, "x": 0.0})

    view = graph_backend.filter().subgraph()
    gc.collect()

    assert len(graph_backend._views) == 1, "referenced view was dropped"

    graph_backend.update_node_attrs(attrs={"x": 2.0}, node_ids=[node_id])
    assert _value_of(view, node_id) == 2.0


@pytest.mark.parametrize("graph_class", [RustWorkXGraph, IndexedRXGraph])
def test_view_registry_survives_pickling(graph_class: type[BaseGraph]) -> None:
    """Pickling a root or a view must not fail on the weak view registry."""
    graph = graph_class()
    graph.add_node_attr_key("x", pl.Float64)
    node_id = graph.add_node({"t": 0, "x": 0.0})
    view = graph.filter().subgraph()

    restored_root = pickle.loads(pickle.dumps(graph))
    assert restored_root.has_node(node_id)
    # a restored root must still be usable for updates
    restored_root.update_node_attrs(attrs={"x": 2.0}, node_ids=[node_id])
    assert _value_of(restored_root, node_id) == 2.0

    restored_view = pickle.loads(pickle.dumps(view))
    assert restored_view.has_node(node_id)


# --------------------------------------------------------------------------
# Root -> view propagation for edge attributes
# --------------------------------------------------------------------------


def _edge_value_of(graph: BaseGraph, edge_id: int, key: str = "w") -> float:
    df = graph.edge_attrs(attr_keys=[DEFAULT_ATTR_KEYS.EDGE_ID, key])
    return df.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_ID) == edge_id)[key].item()


def _graph_with_edge(graph: BaseGraph) -> tuple[int, int]:
    """Add two nodes and an edge between them; return (edge_id, other_edge_id)."""
    graph.add_node_attr_key("x", pl.Float64)
    graph.add_edge_attr_key("w", pl.Float64)
    first = graph.add_node({"t": 0, "x": 0.0})
    second = graph.add_node({"t": 1, "x": 1.0})
    third = graph.add_node({"t": 2, "x": 2.0})
    edge_id = graph.add_edge(first, second, {"w": 0.0})
    other_id = graph.add_edge(second, third, {"w": 0.0})
    return edge_id, other_id


def test_root_edge_update_is_visible_in_view(graph_backend: BaseGraph) -> None:
    """An edge attribute written on the root must be readable through the view."""
    edge_id, _ = _graph_with_edge(graph_backend)

    view = graph_backend.filter().subgraph()

    graph_backend.update_edge_attrs(attrs={"w": 9.0}, edge_ids=[edge_id])

    assert _edge_value_of(graph_backend, edge_id) == 9.0
    assert _edge_value_of(view, edge_id) == 9.0, "view returned a stale edge value after a root write"


def test_root_edge_update_all_edges_is_visible_in_view(graph_backend: BaseGraph) -> None:
    """`edge_ids=None` means all edges, and must reach the view too."""
    edge_id, other_id = _graph_with_edge(graph_backend)

    view = graph_backend.filter().subgraph()

    graph_backend.update_edge_attrs(attrs={"w": 4.0})

    assert _edge_value_of(view, edge_id) == 4.0
    assert _edge_value_of(view, other_id) == 4.0


def test_root_edge_update_per_edge_values_reach_view(graph_backend: BaseGraph) -> None:
    """Per-edge sequence values must land on the matching edges in the view."""
    edge_id, other_id = _graph_with_edge(graph_backend)

    view = graph_backend.filter().subgraph()

    graph_backend.update_edge_attrs(attrs={"w": [11.0, 22.0]}, edge_ids=[edge_id, other_id])

    assert _edge_value_of(view, edge_id) == 11.0
    assert _edge_value_of(view, other_id) == 22.0


def test_view_edge_write_is_applied_once(graph_backend: BaseGraph) -> None:
    """Writing an edge through the view must reach both graphs exactly once."""
    edge_id, _ = _graph_with_edge(graph_backend)

    view = graph_backend.filter().subgraph()

    view.update_edge_attrs(attrs={"w": 3.0}, edge_ids=[edge_id])

    assert _edge_value_of(graph_backend, edge_id) == 3.0
    assert _edge_value_of(view, edge_id) == 3.0


def test_root_edge_update_notifies_multiple_views(graph_backend: BaseGraph) -> None:
    """Every registered view holding the edge must see the new value."""
    edge_id, _ = _graph_with_edge(graph_backend)

    view_a = graph_backend.filter().subgraph()
    view_b = graph_backend.filter().subgraph()

    graph_backend.update_edge_attrs(attrs={"w": 6.0}, edge_ids=[edge_id])

    assert _edge_value_of(view_a, edge_id) == 6.0
    assert _edge_value_of(view_b, edge_id) == 6.0


def test_root_edge_update_partial_overlap_keeps_positions(graph_backend: BaseGraph) -> None:
    """A batch spanning in- and out-of-view edges must apply the right value.

    Per-edge values are positional, so selecting the in-view subset has to keep
    the original indices rather than just the ids.
    """
    graph_backend.add_node_attr_key("x", pl.Float64)
    graph_backend.add_edge_attr_key("w", pl.Float64)
    nodes = [graph_backend.add_node({"t": t, "x": 0.0}) for t in range(4)]
    inside = graph_backend.add_edge(nodes[0], nodes[1], {"w": 0.0})
    outside = graph_backend.add_edge(nodes[2], nodes[3], {"w": 0.0})

    view = graph_backend.filter(NodeAttr("t") <= 1).subgraph()
    assert view.edge_ids() == [inside]

    graph_backend.update_edge_attrs(attrs={"w": [11.0, 22.0]}, edge_ids=[inside, outside])

    assert _edge_value_of(view, inside) == 11.0, "view took the wrong positional value"
    assert _edge_value_of(graph_backend, outside) == 22.0
    assert view._out_of_sync is False
