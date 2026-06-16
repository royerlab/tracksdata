import polars as pl
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph

motile = pytest.importorskip("motile")

from tracksdata.functional import to_motile_graph  # noqa: E402


def _build_graph() -> tuple[RustWorkXGraph, list[int], list[int]]:
    graph = RustWorkXGraph()
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 2, "x": 2.0})

    edge0 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})
    edge1 = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})

    return graph, [node0, node1, node2], [edge0, edge1]


def test_to_motile_graph() -> None:
    graph, node_ids, _ = _build_graph()

    track_graph = to_motile_graph(graph)

    assert isinstance(track_graph, motile.TrackGraph)
    assert set(track_graph.nodes) == set(node_ids)
    assert (node_ids[0], node_ids[1]) in track_graph.edges
    assert (node_ids[1], node_ids[2]) in track_graph.edges

    # frame attribute and node attributes are copied over
    assert track_graph.nodes[node_ids[0]][DEFAULT_ATTR_KEYS.T] == 0
    assert track_graph.nodes[node_ids[2]]["x"] == 2.0
    assert track_graph.get_frames() == (0, 3)

    # edge attributes are copied over, ids/source/target are not added as attributes
    edge_data = track_graph.edges[(node_ids[0], node_ids[1])]
    assert edge_data[DEFAULT_ATTR_KEYS.EDGE_DIST] == -1.0
    assert DEFAULT_ATTR_KEYS.EDGE_ID not in edge_data
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE not in edge_data


def test_to_motile_graph_subset_of_attrs() -> None:
    graph, node_ids, _ = _build_graph()

    track_graph = to_motile_graph(graph, node_attr_keys=[], edge_attr_keys=[])

    # only NODE_ID and frame attribute are kept
    assert set(track_graph.nodes[node_ids[0]]) == {DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.T}
    # edge has no copied attributes
    assert track_graph.edges[(node_ids[0], node_ids[1])] == {}


def test_to_motile_graph_method() -> None:
    graph, node_ids, _ = _build_graph()
    track_graph = graph.to_motile_graph()
    assert isinstance(track_graph, motile.TrackGraph)
    assert set(track_graph.nodes) == set(node_ids)
