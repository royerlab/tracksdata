import numpy as np
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.nodes import RandomNodes
from tracksdata.options import get_options, options_context


def test_random_nodes_init_2d() -> None:
    """Test initialization with 2D coordinates."""
    operator = RandomNodes(n_time_points=5, n_nodes_per_tp=(10, 20), n_dim=2, random_state=42)

    assert operator.n_time_points == 5
    assert operator.n_nodes == (10, 20)
    assert operator.spatial_cols == ["x", "y"]
    assert operator.rng is not None


def test_random_nodes_init_3d() -> None:
    """Test initialization with 3D coordinates."""
    operator = RandomNodes(n_time_points=3, n_nodes_per_tp=(5, 15), n_dim=3, random_state=123)

    assert operator.n_time_points == 3
    assert operator.n_nodes == (5, 15)
    assert operator.spatial_cols == ["x", "y", "z"]


def test_random_nodes_init_invalid_dimension() -> None:
    """Test initialization with invalid dimension raises error."""
    with pytest.raises(ValueError, match="Invalid number of dimensions: 4"):
        RandomNodes(n_time_points=1, n_nodes_per_tp=(1, 2), n_dim=4, random_state=0)


def test_random_nodes_add_nodes_single_time_point_2d() -> None:
    """Test adding nodes for a single time point in 2D."""
    graph = RustWorkXGraph()

    operator = RandomNodes(
        n_time_points=1,
        n_nodes_per_tp=(5, 6),  # Fixed number for deterministic test (low < high)
        n_dim=2,
        random_state=42,
    )

    # Add nodes for time point 0
    operator.add_nodes(graph, t=0)

    # Check that nodes were added
    assert graph.num_nodes == 5  # Should be exactly 5 nodes

    # Check node attributes
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 5
    assert DEFAULT_ATTR_KEYS.T in nodes_df.columns
    assert "x" in nodes_df.columns
    assert "y" in nodes_df.columns
    assert "z" not in nodes_df.columns

    # Check that all nodes have t=0
    assert all(nodes_df[DEFAULT_ATTR_KEYS.T] == 0)

    # Check that coordinates are in [0, 1] range
    assert all(0 <= x <= 1 for x in nodes_df["x"])
    assert all(0 <= y <= 1 for y in nodes_df["y"])


def test_random_nodes_add_nodes_single_time_point_3d() -> None:
    """Test adding nodes for a single time point in 3D."""
    graph = RustWorkXGraph()

    operator = RandomNodes(
        n_time_points=1,
        n_nodes_per_tp=(3, 4),  # Fixed number for deterministic test (low < high)
        n_dim=3,
        random_state=42,
    )

    # Add nodes for time point 1
    operator.add_nodes(graph, t=1)

    # Check that nodes were added
    assert graph.num_nodes == 3  # Should be exactly 3 nodes

    # Check node attributes
    nodes_df = graph.node_attrs()
    assert len(nodes_df) == 3
    assert DEFAULT_ATTR_KEYS.T in nodes_df.columns
    assert "x" in nodes_df.columns
    assert "y" in nodes_df.columns
    assert "z" in nodes_df.columns

    # Check that all nodes have t=1
    assert all(nodes_df[DEFAULT_ATTR_KEYS.T] == 1)

    # Check that coordinates are in [0, 1] range
    assert all(0 <= x <= 1 for x in nodes_df["x"])
    assert all(0 <= y <= 1 for y in nodes_df["y"])
    assert all(0 <= z <= 1 for z in nodes_df["z"])


@pytest.mark.parametrize("n_workers", [1, 2])
def test_random_nodes_add_nodes_all_time_points(n_workers: int) -> None:
    """Test adding nodes for all time points when t=None with different worker counts."""
    graph = RustWorkXGraph()

    operator = RandomNodes(
        n_time_points=3,
        n_nodes_per_tp=(2, 5),
        n_dim=2,
        random_state=42,  # Range where low < high
    )

    # Add nodes for all time points
    with options_context(n_workers=n_workers):
        operator.add_nodes(graph)

    # Check that nodes were added for all time points
    nodes_df = graph.node_attrs()
    time_points = sorted(nodes_df[DEFAULT_ATTR_KEYS.T].unique())
    assert time_points == [0, 1, 2]

    # Check that each time point has between 2-4 nodes
    for t in time_points:
        nodes_at_t = nodes_df.filter(nodes_df[DEFAULT_ATTR_KEYS.T] == t)
        assert 2 <= len(nodes_at_t) < 5


def test_random_nodes_variable_node_count() -> None:
    """Test that the number of nodes varies within the specified range."""
    graph = RustWorkXGraph()

    operator = RandomNodes(
        n_time_points=10,
        n_nodes_per_tp=(1, 11),
        n_dim=2,
        random_state=42,  # Range where low < high
    )

    # Add nodes for all time points
    operator.add_nodes(graph)

    # Check that different time points have different numbers of nodes
    nodes_df = graph.node_attrs()
    node_counts = []
    for t in range(10):
        nodes_at_t = nodes_df.filter(nodes_df[DEFAULT_ATTR_KEYS.T] == t)
        node_counts.append(len(nodes_at_t))
        assert 1 <= len(nodes_at_t) < 11

    # Should have some variation in node counts
    assert len(set(node_counts)) > 1


def test_random_nodes_reproducible_with_same_seed() -> None:
    """Test that results are reproducible with the same random seed."""
    # First run
    graph1 = RustWorkXGraph()
    operator1 = RandomNodes(
        n_time_points=2,
        n_nodes_per_tp=(3, 4),
        n_dim=2,
        random_state=123,  # Range where low < high
    )
    operator1.add_nodes(graph1)
    nodes_df1 = graph1.node_attrs()

    # Second run with same seed
    graph2 = RustWorkXGraph()
    operator2 = RandomNodes(
        n_time_points=2,
        n_nodes_per_tp=(3, 4),
        n_dim=2,
        random_state=123,  # Range where low < high
    )
    operator2.add_nodes(graph2)
    nodes_df2 = graph2.node_attrs()

    # Results should be identical
    assert len(nodes_df1) == len(nodes_df2)

    # Sort by time and then by coordinates for consistent comparison
    nodes_df1_sorted = nodes_df1.sort([DEFAULT_ATTR_KEYS.T, "x", "y"])
    nodes_df2_sorted = nodes_df2.sort([DEFAULT_ATTR_KEYS.T, "x", "y"])

    # Compare coordinates (allowing for small floating point differences)
    for col in ["x", "y"]:
        coords1 = nodes_df1_sorted[col].to_numpy()
        coords2 = nodes_df2_sorted[col].to_numpy()
        assert np.allclose(coords1, coords2)


def test_random_nodes_different_with_different_seed() -> None:
    """Test that results are different with different random seeds."""
    # First run
    graph1 = RustWorkXGraph()
    operator1 = RandomNodes(
        n_time_points=1,
        n_nodes_per_tp=(5, 6),
        n_dim=2,
        random_state=42,  # Range where low < high
    )
    operator1.add_nodes(graph1, t=0)
    nodes_df1 = graph1.node_attrs()

    # Second run with different seed
    graph2 = RustWorkXGraph()
    operator2 = RandomNodes(
        n_time_points=1,
        n_nodes_per_tp=(5, 6),
        n_dim=2,
        random_state=999,  # Range where low < high
    )
    operator2.add_nodes(graph2, t=0)
    nodes_df2 = graph2.node_attrs()

    # Results should be different
    coords1 = nodes_df1["x"].to_numpy()
    coords2 = nodes_df2["x"].to_numpy()
    assert not np.allclose(coords1, coords2)


def test_random_nodes_attr_keys_registration() -> None:
    """Test that spatial attribute keys are properly registered."""
    graph = RustWorkXGraph()

    operator = RandomNodes(
        n_time_points=1,
        n_nodes_per_tp=(2, 3),
        n_dim=3,
        random_state=42,  # Range where low < high
    )

    # Initially, spatial keys should not be registered
    assert "x" not in graph.node_attr_keys
    assert "y" not in graph.node_attr_keys
    assert "z" not in graph.node_attr_keys

    # Add nodes
    operator.add_nodes(graph)

    # Now spatial keys should be registered
    assert "x" in graph.node_attr_keys
    assert "y" in graph.node_attr_keys
    assert "z" in graph.node_attr_keys


def test_random_nodes_empty_time_points() -> None:
    """Test behavior with zero time points."""
    graph = RustWorkXGraph()

    operator = RandomNodes(n_time_points=0, n_nodes_per_tp=(1, 5), n_dim=2, random_state=42)

    # Add nodes for all time points (should be none)
    operator.add_nodes(graph)

    # Graph should remain empty
    assert graph.num_nodes == 0


def test_random_nodes_multiprocessing_isolation() -> None:
    """Test that multiprocessing options don't affect subsequent tests."""
    # Verify default n_workers is 1
    assert get_options().n_workers == 1
