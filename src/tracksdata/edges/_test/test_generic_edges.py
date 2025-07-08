import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import GenericFuncEdgeAttrs
from tracksdata.graph import RustWorkXGraph


def _scalar_distance_func(source_val: float, target_val: float) -> float:
    """
    Compute the absolute distance between two scalar values.
    """
    return abs(source_val - target_val)


def test_generic_edges_init_single_attr_key() -> None:
    """Test initialization with single attribute key."""
    operator = GenericFuncEdgeAttrs(func=_scalar_distance_func, attr_keys="x", output_key="distance")

    assert operator.attr_keys == "x"
    assert operator.func == _scalar_distance_func
    assert operator.output_key == "distance"


def test_generic_edges_init_multiple_attr_keys() -> None:
    """Test initialization with multiple attribute keys."""

    def _euclidean_distance(source_attrs, target_attrs):
        dx = source_attrs["x"] - target_attrs["x"]
        dy = source_attrs["y"] - target_attrs["y"]
        return np.sqrt(dx**2 + dy**2)

    operator = GenericFuncEdgeAttrs(func=_euclidean_distance, attr_keys=["x", "y"], output_key="euclidean_distance")

    assert operator.attr_keys == ["x", "y"]
    assert operator.func == _euclidean_distance
    assert operator.output_key == "euclidean_distance"


def test_generic_edges_add_weights_single_attr_key() -> None:
    """Test adding weights with single attribute key."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes at time 0
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 4.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 7.0})

    # Add edges
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})
    edge2 = graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    operator = GenericFuncEdgeAttrs(func=_scalar_distance_func, attr_keys="x", output_key="distance")

    operator.add_edge_attrs(graph)

    # Check that weights were added
    edges_df = graph.edge_attrs()

    assert len(edges_df) == 2
    assert "distance" in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_WEIGHT in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns

    # Check specific distances
    edge_distances = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["distance"], strict=False))
    assert edge_distances[edge1] == 3.0  # |1.0 - 4.0|
    assert edge_distances[edge2] == 3.0  # |4.0 - 7.0|


def test_generic_edges_add_weights_multiple_attr_keys() -> None:
    """Test adding weights with multiple attribute keys."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes at time 0
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 3.0, "y": 4.0})

    # Add edge
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    def euclidean_distance(source_attrs, target_attrs):
        dx = source_attrs["x"] - target_attrs["x"]
        dy = source_attrs["y"] - target_attrs["y"]
        return np.sqrt(dx**2 + dy**2)

    operator = GenericFuncEdgeAttrs(func=euclidean_distance, attr_keys=["x", "y"], output_key="euclidean_distance")

    operator.add_edge_attrs(graph, t=0)

    # Check that weights were added
    edges_df = graph.edge_attrs()
    assert "euclidean_distance" in edges_df.columns

    # Check specific distance (3-4-5 triangle)
    edge_distances = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["euclidean_distance"], strict=False))
    assert abs(edge_distances[edge1] - 5.0) < 1e-6


def test_generic_edges_add_weights_all_time_points() -> None:
    """Test adding weights to all time points when t=None."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes at different time points
    node0_t0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0})
    node1_t0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 2.0})
    node0_t1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 3.0})
    node1_t1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 6.0})

    # Add edges from t=0 to t=1 (temporal edges)
    edge_t0_to_t1_1 = graph.add_edge(node0_t0, node0_t1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})
    edge_t0_to_t1_2 = graph.add_edge(node1_t0, node1_t1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    operator = GenericFuncEdgeAttrs(func=_scalar_distance_func, attr_keys="x", output_key="distance")

    # Add weights to all time points
    operator.add_edge_attrs(graph)

    # Check that weights were added to both edges
    edges_df = graph.edge_attrs()
    edge_distances = dict(zip(edges_df[DEFAULT_ATTR_KEYS.EDGE_ID], edges_df["distance"], strict=False))
    assert edge_distances[edge_t0_to_t1_1] == 2.0  # |1.0 - 3.0|
    assert edge_distances[edge_t0_to_t1_2] == 4.0  # |2.0 - 6.0|


def test_generic_edges_no_edges_at_time_point() -> None:
    """Test behavior when no edges exist at a time point."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)

    # Add nodes but no edges at time 0
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 2.0})

    operator = GenericFuncEdgeAttrs(func=_scalar_distance_func, attr_keys="x", output_key="distance")

    # This should not raise an error, just log a warning
    operator.add_edge_attrs(graph)

    # Verify no edges exist
    edges_df = graph.edge_attrs()
    assert len(edges_df) == 0


def test_generic_edges_creates_output_key() -> None:
    """Test that the operator creates the output key if it doesn't exist."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes and edge
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 4.0})
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    operator = GenericFuncEdgeAttrs(func=_scalar_distance_func, attr_keys="x", output_key="new_distance_key")

    # Verify the key doesn't exist initially
    assert "new_distance_key" not in graph.edge_attr_keys

    operator.add_edge_attrs(graph)

    # Verify the key was created
    assert "new_distance_key" in graph.edge_attr_keys

    # Verify the weights were added
    edges_df = graph.edge_attrs()
    assert "new_distance_key" in edges_df.columns


def test_generic_edges_dict_input_function() -> None:
    """Test with a more complex function that uses multiple operations."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("value", 0.0)
    graph.add_node_attr_key("weight", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_WEIGHT, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 10.0, "weight": 2.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "value": 20.0, "weight": 3.0})

    # Add edge
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_WEIGHT: 0.0})

    def weighted_difference(source_attrs: dict[str, float], target_attrs: dict[str, float]) -> float:
        diff = abs(source_attrs["value"] - target_attrs["value"])
        avg_weight = (source_attrs["weight"] + target_attrs["weight"]) / 2
        return diff * avg_weight

    operator = GenericFuncEdgeAttrs(func=weighted_difference, attr_keys=["value", "weight"], output_key="weighted_diff")

    operator.add_edge_attrs(graph)

    # Check the computed weight
    edges_df = graph.edge_attrs()
    expected_weight = abs(10.0 - 20.0) * ((2.0 + 3.0) / 2)  # 10 * 2.5 = 25.0
    assert abs(edges_df["weighted_diff"][0] - expected_weight) < 1e-6


def test_generic_edges_empty_graph() -> None:
    """Test behavior with an empty graph."""
    graph = RustWorkXGraph()

    operator = GenericFuncEdgeAttrs(func=_scalar_distance_func, attr_keys="x", output_key="distance")

    # This should not raise an error
    operator.add_edge_attrs(graph, t=0)

    # Verify graph is still empty
    assert graph.num_nodes == 0
    assert graph.num_edges == 0
