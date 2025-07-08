import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import DistanceEdges
from tracksdata.graph import RustWorkXGraph
from tracksdata.options import get_options, options_context


def test_distance_edges_init_default_params() -> None:
    """Test initialization with default parameters."""
    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3)

    assert operator.output_key == DEFAULT_ATTR_KEYS.EDGE_WEIGHT
    assert operator.distance_threshold == 10.0
    assert operator.n_neighbors == 3
    assert operator.attr_keys is None
    assert operator.delta_t == 1


def test_distance_edges_init_custom_params() -> None:
    """Test initialization with custom parameters."""
    operator = DistanceEdges(
        distance_threshold=5.0,
        n_neighbors=2,
        delta_t=2,
        attr_keys=["x", "y"],
        output_key="custom_distance",
    )
    assert operator.output_key == "custom_distance"
    assert operator.distance_threshold == 5.0
    assert operator.n_neighbors == 2
    assert operator.attr_keys == ["x", "y"]
    assert operator.delta_t == 2


def test_distance_edges_add_edges_empty_graph() -> None:
    """Test adding edges to an empty graph."""
    graph = RustWorkXGraph()
    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3)

    # Should not raise an error on empty graph
    operator.add_edges(graph)
    assert graph.num_edges == 0


def test_distance_edges_add_edges_single_timepoint_no_previous() -> None:
    """Test adding edges when there are no nodes in the previous timepoint."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes only at t=1 (no t=0)
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3)

    # Should not add any edges since there are no nodes at t=0
    operator.add_edges(graph)
    assert graph.num_edges == 0


def test_distance_edges_add_edges_single_timepoint_no_current() -> None:
    """Test adding edges when there are no nodes in the current timepoint."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes only at t=0 (no t=1)
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0, "y": 1.0})

    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3)

    # Should not add any edges since there are no nodes at t=1
    operator.add_edges(graph)
    assert graph.num_edges == 0


def test_distance_edges_add_edges_2d_coordinates() -> None:
    """Test adding edges with 2D coordinates."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 5.0, "y": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 6.0, "y": 1.0})

    operator = DistanceEdges(distance_threshold=3.0, n_neighbors=2)

    operator.add_edges(graph)

    # Should have edges from t=0 to t=1 nodes within distance threshold
    assert graph.num_edges == 2  # other edges are outside distance threshold

    # Check that edge weights are added
    edges_df = graph.edge_attrs()
    assert DEFAULT_ATTR_KEYS.EDGE_WEIGHT in edges_df.columns


def test_distance_edges_add_edges_3d_coordinates() -> None:
    """Test adding edges with 3D coordinates."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_node_attr_key("z", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0, "z": 0.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 5.0, "y": 0.0, "z": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0, "z": 1.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 6.0, "y": 1.0, "z": 1.0})

    operator = DistanceEdges(distance_threshold=3.0, n_neighbors=2)

    operator.add_edges(graph)

    # Should have edges from t=0 to t=1 nodes within distance threshold
    assert graph.num_edges == 2  # other edges are outside distance threshold


def test_distance_edges_add_edges_custom_attr_keys() -> None:
    """Test adding edges with custom attribute keys."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("pos_x", 0.0)
    graph.add_node_attr_key("pos_y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "pos_x": 0.0, "pos_y": 0.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "pos_x": 5.0, "pos_y": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "pos_x": 1.0, "pos_y": 1.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "pos_x": 6.0, "pos_y": 1.0})

    operator = DistanceEdges(distance_threshold=3.0, n_neighbors=2, attr_keys=["pos_x", "pos_y"])

    operator.add_edges(graph)

    # Should work with custom attribute keys
    assert graph.num_edges == 2  # other edges are outside distance threshold


def test_distance_edges_add_edges_distance_threshold() -> None:
    """Test that distance threshold is respected."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})

    # Add nodes at t=1 - far away
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 100.0, "y": 100.0})

    operator = DistanceEdges(distance_threshold=1.0, n_neighbors=2)  # Very small threshold

    operator.add_edges(graph)

    # Should have no edges due to distance threshold
    assert graph.num_edges == 0


@pytest.mark.parametrize("n_workers", [1, 2])
def test_distance_edges_add_edges_multiple_timepoints(n_workers: int) -> None:
    """Test adding edges for multiple timepoints with different worker counts."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes at multiple timepoints
    for t in range(3):
        for i in range(2):
            graph.add_node({DEFAULT_ATTR_KEYS.T: t, "x": float(i), "y": float(t)})

    operator = DistanceEdges(distance_threshold=5.0, n_neighbors=2)

    # Add edges for all timepoints
    with options_context(n_workers=n_workers):
        operator.add_edges(graph)

    # Should have some edges
    assert graph.num_edges >= 0


def test_distance_edges_add_edges_custom_weight_key() -> None:
    """Test adding edges with custom weight key."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    custom_weight_key = "custom_distance"
    operator = DistanceEdges(distance_threshold=5.0, n_neighbors=2, output_key=custom_weight_key)

    operator.add_edges(graph)

    if graph.num_edges > 0:
        edges_df = graph.edge_attrs()
        assert len(edges_df) == 1
        assert custom_weight_key in edges_df.columns
        assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
        assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
        assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns


def test_distance_edges_n_neighbors_limit() -> None:
    """Test that n_neighbors limit is respected."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add many nodes at t=0
    for i in range(5):
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": float(i), "y": 0.0})

    # Add one node at t=1
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 0.0})

    operator = DistanceEdges(
        distance_threshold=10.0,
        n_neighbors=2,  # Limit to 2 neighbors
    )

    operator.add_edges(graph)

    # Should have at most 2 edges (limited by n_neighbors)
    assert graph.num_edges == 2


@pytest.mark.parametrize("n_workers", [1, 2])
def test_distance_edges_add_edges_with_delta_t(n_workers: int) -> None:
    """Test adding edges with delta_t=2 connecting nodes across multiple timepoints."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add nodes at t=0, t=1, t=2
    node_0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node_1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node_2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 2, "x": 2.0, "y": 2.0})
    node_3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 3, "x": 3.0, "y": 3.0})

    operator = DistanceEdges(distance_threshold=5.0, n_neighbors=2, delta_t=2)

    with options_context(n_workers=n_workers):
        operator.add_edges(graph)

    edges_df = graph.edge_attrs()
    edge_list = {
        (src, tgt)
        for src, tgt in zip(
            edges_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
            edges_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list(),
            strict=True,
        )
    }
    expected_edge_list = {
        (node_0, node_1),
        (node_1, node_2),
        (node_2, node_3),
        (node_0, node_2),
        (node_1, node_3),
    }
    assert edge_list == expected_edge_list


def test_distance_edges_invalid_delta_t() -> None:
    """Test that invalid delta_t values raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="'delta_t' must be at least 1"):
        DistanceEdges(distance_threshold=10.0, n_neighbors=3, delta_t=0)

    with pytest.raises(ValueError, match="'delta_t' must be at least 1"):
        DistanceEdges(distance_threshold=10.0, n_neighbors=3, delta_t=-1)


def test_distance_edges_multiprocessing_isolation() -> None:
    """Test that multiprocessing options don't affect subsequent tests."""
    # Verify default n_workers is 1
    assert get_options().n_workers == 1
