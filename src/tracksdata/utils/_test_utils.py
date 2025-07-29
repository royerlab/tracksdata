"""
Utility functions for setting up tests to reduce code duplication.
"""

from typing import Any

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph


def setup_mask_attrs(graph: BaseGraph) -> None:
    """Set up mask and bbox attribute keys on a graph."""
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)


def setup_spatial_attrs_2d(graph: BaseGraph, default_value: float = 0.0) -> None:
    """Set up 2D spatial coordinate attributes (x, y) on a graph."""
    graph.add_node_attr_key("x", default_value)
    graph.add_node_attr_key("y", default_value)


def setup_spatial_attrs_3d(graph: BaseGraph, default_value: float = 0.0) -> None:
    """Set up 3D spatial coordinate attributes (x, y, z) on a graph."""
    setup_spatial_attrs_2d(graph, default_value)
    graph.add_node_attr_key("z", default_value)


def setup_time_attr(graph: BaseGraph, default_value: int = 0) -> None:
    """Set up time attribute on a graph."""
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.T, default_value)


def setup_edge_distance_attr(graph: BaseGraph, default_value: float = 0.0) -> None:
    """Set up edge distance attribute on a graph."""
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, default_value)


def setup_solution_attrs(graph: BaseGraph) -> None:
    """Set up solution attributes for tracking on a graph."""
    graph.add_node_attr_key(DEFAULT_ATTR_KEYS.SOLUTION, True)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.SOLUTION, True)


def setup_tracking_graph(
    graph: BaseGraph,
    *,
    spatial_dims: int = 2,
    include_time: bool = True,
    include_mask: bool = False,
    include_edge_dist: bool = True,
    spatial_default: float = 0.0,
    time_default: int = 0,
) -> None:
    """
    Set up a graph with common tracking attributes.

    Parameters
    ----------
    graph : BaseGraph
        The graph to set up.
    spatial_dims : int, default 2
        Number of spatial dimensions (2 or 3).
    include_time : bool, default True
        Whether to include time attribute.
    include_mask : bool, default False
        Whether to include mask and bbox attributes.
    include_edge_dist : bool, default True
        Whether to include edge distance attribute.
    spatial_default : float, default 0.0
        Default value for spatial coordinates.
    time_default : int, default 0
        Default value for time attribute.
    """
    if include_time:
        setup_time_attr(graph, time_default)

    if spatial_dims == 2:
        setup_spatial_attrs_2d(graph, spatial_default)
    elif spatial_dims == 3:
        setup_spatial_attrs_3d(graph, spatial_default)
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    if include_mask:
        setup_mask_attrs(graph)

    if include_edge_dist:
        setup_edge_distance_attr(graph)


def setup_custom_node_attr(graph: BaseGraph, key: str, default_value: Any) -> None:
    """Set up a custom node attribute on a graph."""
    graph.add_node_attr_key(key, default_value)


def setup_custom_edge_attr(graph: BaseGraph, key: str, default_value: Any) -> None:
    """Set up a custom edge attribute on a graph."""
    graph.add_edge_attr_key(key, default_value)
