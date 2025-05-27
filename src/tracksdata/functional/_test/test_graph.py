import numpy as np
import pytest
import rustworkx as rx

from tracksdata.functional.graph import graph_track_ids


def test_empty_graph():
    """Test that empty graph raises ValueError."""
    graph = rx.PyDiGraph()
    with pytest.raises(ValueError, match="Graph is empty"):
        graph_track_ids(graph)


def test_single_path():
    """Test graph with a single linear path."""
    graph = rx.PyDiGraph()

    # Add nodes 0->1->2 (single parent, single child)
    nodes = [graph.add_node(None) for _ in range(3)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[1], nodes[2], None)

    node_ids, track_ids, tracks_graph = graph_track_ids(graph)

    assert np.array_equal(node_ids, [0, 1, 2])
    assert np.array_equal(track_ids, [1, 1, 1])
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 1 + 1  # Single track (includes null node (0))


def test_branching_path():
    """Test graph with a valid branching path (two children)."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #     0
    #    / \
    #   1   2
    nodes = [graph.add_node(None) for _ in range(3)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[0], nodes[2], None)

    node_ids, track_ids, tracks_graph = graph_track_ids(graph)

    # Should create 2 tracks: one for each branch
    assert len(node_ids) == 3
    assert len(track_ids) == 3
    assert len(np.unique(track_ids)) == 3  # Three unique track IDs
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 3 + 1  # Three tracks (includes null node (0))


def test_invalid_multiple_parents():
    """Test graph with invalid structure (node with multiple parents, merge)."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #   0   1
    #    \ /
    #     2
    nodes = [graph.add_node(None) for _ in range(3)]
    graph.add_edge(nodes[0], nodes[2], None)
    graph.add_edge(nodes[1], nodes[2], None)

    with pytest.raises(RuntimeError, match="Invalid graph structure"):
        graph_track_ids(graph)


def test_complex_valid_branching():
    """Test graph with complex but valid branching pattern."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #     0
    #    / \
    #   1   2
    #  /     \
    # 3       4
    nodes = [graph.add_node(None) for _ in range(5)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[0], nodes[2], None)
    graph.add_edge(nodes[1], nodes[3], None)
    graph.add_edge(nodes[2], nodes[4], None)

    node_ids, track_ids, tracks_graph = graph_track_ids(graph)

    assert len(node_ids) == 5
    assert len(track_ids) == 5
    assert len(np.unique(track_ids)) == 3  # Five unique track IDs
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 3 + 1  # Five tracks (includes null node (0))


def test_invalid_three_children():
    """Test graph with invalid structure (node with 3 children)."""
    graph = rx.PyDiGraph()

    # Add nodes:
    #     0
    #   / | \
    #  1  2  3
    nodes = [graph.add_node(None) for _ in range(4)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[0], nodes[2], None)
    graph.add_edge(nodes[0], nodes[3], None)

    with pytest.raises(RuntimeError, match="Invalid graph structure"):
        graph_track_ids(graph)


def test_multiple_roots():
    """Test graph with multiple valid root nodes."""
    graph = rx.PyDiGraph()

    # Add nodes:
    # 0->1  2->3
    nodes = [graph.add_node(None) for _ in range(4)]
    graph.add_edge(nodes[0], nodes[1], None)
    graph.add_edge(nodes[2], nodes[3], None)

    node_ids, track_ids, tracks_graph = graph_track_ids(graph)

    assert len(node_ids) == 4
    assert len(track_ids) == 4
    assert len(np.unique(track_ids)) == 2  # Two unique track IDs
    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 2 + 1  # Two separate tracks (includes null node (0))
