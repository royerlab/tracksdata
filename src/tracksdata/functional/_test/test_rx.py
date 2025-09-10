import numpy as np
import pytest
import rustworkx as rx

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import _assign_track_ids


def test_empty_graph() -> None:
    """Test that empty graph raises ValueError."""
    graph = rx.PyDiGraph()
    with pytest.raises(ValueError, match="Graph is empty"):
        _assign_track_ids(graph, track_id_offset=1)


def test_single_path() -> None:
    """Test graph with a single linear path."""
    graph = rx.PyDiGraph()

    # Add nodes 0->1->2 (single parent, single child)
    nodes = [graph.add_node({DEFAULT_ATTR_KEYS.T: t}) for t in range(3)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[1], nodes[2], None)

    node_ids, track_ids, tracks_graph = _assign_track_ids(graph, track_id_offset=1)

    assert np.array_equal(node_ids, [0, 1, 2])
    assert np.array_equal(track_ids, [1, 1, 1])
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 1  # Single track


def test_symmetric_branching_path() -> None:
    """Test graph with a valid branching path (two children)."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #     0
    #    / \
    #   1   2
    nodes = [
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
    ]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[0], nodes[2], None)

    node_ids, track_ids, tracks_graph = _assign_track_ids(graph, track_id_offset=1)

    # Should create 2 tracks: one for each branch
    assert len(node_ids) == 3
    assert len(track_ids) == 3
    assert len(np.unique(track_ids)) == 3  # Three unique track IDs
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 3  # Three tracks


def test_asymmetric_branching_path() -> None:
    """Test graph with a valid branching path (two children)."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #     0
    #    / \
    #   1   \
    #  / .   \
    # 2 .     3
    nodes = [
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 2}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 2}),
    ]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[1], nodes[2], None)
    graph.add_edge(nodes[0], nodes[3], None)

    node_ids, track_ids, tracks_graph = _assign_track_ids(graph, track_id_offset=1)

    # Should create 2 tracks: one for each branch
    assert len(node_ids) == 4
    assert len(track_ids) == 4
    assert len(np.unique(track_ids)) == 3  # Three unique track IDs
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 3  # Three tracks


def test_invalid_multiple_parents() -> None:
    """Test graph with invalid structure (node with multiple parents, merge)."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #   0   1
    #    \ /
    #     2
    nodes = [
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
    ]
    graph.add_edge(nodes[0], nodes[2], None)
    graph.add_edge(nodes[1], nodes[2], None)

    with pytest.raises(RuntimeError, match="Invalid graph structure"):
        _assign_track_ids(graph, track_id_offset=1)


def test_complex_valid_branching() -> None:
    """Test graph with complex but valid branching pattern."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #       0
    #      / \
    #     1   2
    #    /     \
    #   3      |
    #    \     /
    #     4   5
    nodes = [
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 2}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 3}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 3}),
    ]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[0], nodes[2], None)
    graph.add_edge(nodes[1], nodes[3], None)
    graph.add_edge(nodes[3], nodes[4], None)
    graph.add_edge(nodes[2], nodes[5], None)

    node_ids, track_ids, tracks_graph = _assign_track_ids(graph, track_id_offset=1)

    # this order is an implementation detail, it could change
    # then the track ids should change accordingly
    np.testing.assert_array_equal(node_ids, [0, 1, 3, 4, 2, 5])
    np.testing.assert_array_equal(track_ids, [1, 2, 2, 2, 3, 4])

    assert set(tracks_graph.successor_indices(tracks_graph.find_node_by_weight(1))) == set(
        map(tracks_graph.find_node_by_weight, {2, 3})
    )
    assert set(tracks_graph.successor_indices(tracks_graph.find_node_by_weight(3))) == set(
        map(tracks_graph.find_node_by_weight, {4})
    )
    assert tracks_graph.num_edges() == 3

    assert len(node_ids) == 6
    assert len(track_ids) == 6
    assert len(np.unique(track_ids)) == 4  # {0, {1, 3, 4}, 2, 5}
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 4  # Four tracks


def test_three_children() -> None:
    """Test graph with 3 children."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #     0
    #   / | \
    #  1  2  3
    nodes = [
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        graph.add_node({DEFAULT_ATTR_KEYS.T: 1}),
    ]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[0], nodes[2], None)
    graph.add_edge(nodes[0], nodes[3], None)

    _, track_ids, tracks_graph = _assign_track_ids(graph, track_id_offset=1)
    track_graphs_node_id = tracks_graph.find_node_by_weight(track_ids[0])
    successor_node_ids = tracks_graph.successor_indices(track_graphs_node_id)
    assert {tracks_graph[i] for i in successor_node_ids} == set(track_ids[1:])


def test_multiple_roots() -> None:
    """Test graph with multiple valid root nodes."""
    graph = rx.PyDiGraph()

    # Add nodes:
    # 0->1  2->3
    nodes = [graph.add_node({DEFAULT_ATTR_KEYS.T: t % 2}) for t in range(4)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[2], nodes[3], None)

    node_ids, track_ids, tracks_graph = _assign_track_ids(graph, track_id_offset=1)

    assert len(node_ids) == 4
    assert len(track_ids) == 4
    assert len(np.unique(track_ids)) == 2  # Two unique track IDs
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 2  # Two separate tracks
