import polars as pl
import pytest

from tracksdata.attrs import Attr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.solvers import NearestNeighborsSolver


def test_nearest_neighbors_solver_init_default() -> None:
    """Test NearestNeighborsSolver initialization with default parameters."""
    solver = NearestNeighborsSolver()

    assert solver.max_children == 2
    assert solver.output_key == DEFAULT_ATTR_KEYS.SOLUTION
    assert isinstance(solver.edge_weight_expr, Attr)


def test_nearest_neighbors_solver_init_custom() -> None:
    """Test NearestNeighborsSolver initialization with custom parameters."""
    solver = NearestNeighborsSolver(max_children=3, edge_weight="custom_weight", output_key="custom_solution")

    assert solver.max_children == 3
    assert solver.output_key == "custom_solution"
    assert isinstance(solver.edge_weight_expr, Attr)


def test_nearest_neighbors_solver_init_with_attr_expr() -> None:
    """Test NearestNeighborsSolver initialization with AttrExpr."""
    weight_expr = Attr("weight") * 2
    solver = NearestNeighborsSolver(edge_weight=weight_expr)

    assert str(solver.edge_weight_expr) == str(weight_expr)


def test_nearest_neighbors_solver_solve_empty_graph() -> None:
    """Test solving on an empty graph."""
    graph = RustWorkXGraph()
    solver = NearestNeighborsSolver()

    # Should not raise an error on empty graph
    with pytest.raises(ValueError):
        solver.solve(graph)


def test_nearest_neighbors_solver_solve_no_edges() -> None:
    """Test solving on a graph with nodes but no edges."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)

    # Add some nodes
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    solver = NearestNeighborsSolver()

    # Should not raise an error with no edges
    with pytest.raises(ValueError):
        solver.solve(graph)


def test_nearest_neighbors_solver_solve_simple_case() -> None:
    """Test solving with a simple graph."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges with weights
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})
    edge2 = graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 2.0})

    solver = NearestNeighborsSolver(max_children=1)
    solver.solve(graph)

    # Check that solution keys are added
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_attrs.columns

    solutions = edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION].to_list()
    edge_ids = edge_attrs[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

    # Check that the edges are selected
    assert [True, False] == solutions
    assert [edge1, edge2] == edge_ids


def test_nearest_neighbors_solver_solve_max_children_constraint() -> None:
    """Test that max_children constraint is respected."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})  # Parent
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})  # Child 1
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})  # Child 2
    node3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 3.0, "y": 3.0})  # Child 3

    # Add edges with different weights (lower weight = better)
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})  # Best
    edge2 = graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 2.0})  # Second best
    _ = graph.add_edge(node0, node3, {DEFAULT_ATTR_KEYS.EDGE_DIST: 3.0})  # Worst

    solver = NearestNeighborsSolver(max_children=2)
    solver.solve(graph)

    # Check that only 2 edges are selected (max_children constraint)
    edge_attrs = graph.edge_attrs()
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_edges) == 2

    # Check that the best 2 edges are selected
    selected_weights = selected_edges[DEFAULT_ATTR_KEYS.EDGE_DIST].to_list()

    assert 1.0 in selected_weights
    assert 2.0 in selected_weights
    assert 3.0 not in selected_weights

    selected_ids = selected_edges[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
    assert [edge1, edge2] == selected_ids


def test_nearest_neighbors_solver_solve_one_parent_constraint() -> None:
    """Test that each node can have only one parent."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})  # Parent 1
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0, "y": 0.0})  # Parent 2
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 0.5, "y": 1.0})  # Child

    # Add edges to the same child from different parents
    graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})  # Better edge
    graph.add_edge(node1, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: 2.0})  # Worse edge

    solver = NearestNeighborsSolver()
    solver.solve(graph)

    # Check that only one edge to the child is selected
    edge_attrs = graph.edge_attrs()
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    child_edges = selected_edges.filter(selected_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET] == node2)

    assert len(child_edges) == 1

    # Check that the better edge is selected
    assert child_edges[DEFAULT_ATTR_KEYS.EDGE_DIST].to_list()[0] == 1.0


def test_nearest_neighbors_solver_solve_custom_weight_expr() -> None:
    """Test solving with custom weight expression."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key("custom_weight", dtype=pl.Float64)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges with custom weight attribute
    edge1 = graph.add_edge(node0, node1, {"custom_weight": 2.0})
    graph.add_edge(node0, node2, {"custom_weight": 1.0})

    # Use negative custom weight (so lower values become higher priority)
    weight_expr = -Attr("custom_weight")
    solver = NearestNeighborsSolver(edge_weight=weight_expr, max_children=1)
    solver.solve(graph)

    # Check that the edge with higher custom_weight (2.0) is selected
    # because we're using negative weight
    edge_attrs = graph.edge_attrs()
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_ids = selected_edges[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

    assert len(selected_edges) == 1
    assert [edge1] == selected_ids


def test_nearest_neighbors_solver_solve_complex_expression() -> None:
    """Test solving with complex weight expression."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key("distance", dtype=pl.Float64)
    graph.add_edge_attr_key("confidence", dtype=pl.Float64)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges with multiple attributes
    graph.add_edge(node0, node1, {"distance": 1.0, "confidence": 0.8})
    graph.add_edge(node0, node2, {"distance": 2.0, "confidence": 0.9})

    # Use complex expression: distance / confidence (lower is better)
    weight_expr = Attr("distance") / Attr("confidence")
    solver = NearestNeighborsSolver(edge_weight=weight_expr, max_children=1)
    solver.solve(graph)

    # Check that the edge with better distance/confidence ratio is selected
    edge_attrs = graph.edge_attrs()
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_edges) == 1
    # Edge 0->1: 1.0/0.8 = 1.25
    # Edge 0->2: 2.0/0.9 = 2.22
    # Edge 0->1 should be selected (lower ratio)
    assert selected_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[0] == node1


def test_nearest_neighbors_solver_solve_custom_output_key() -> None:
    """Test solving with custom output key."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    # Add nodes and edges
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: 1.0})

    custom_key = "my_solution"
    solver = NearestNeighborsSolver(output_key=custom_key)
    solver.solve(graph)

    # Check that custom key is used
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert custom_key in node_attrs.columns
    assert custom_key in edge_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION not in node_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION not in edge_attrs.columns


def test_nearest_neighbors_solver_solve_with_overlaps() -> None:
    """Test solving with overlapping nodes that should be mutually exclusive."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    # Add nodes - overlapping pair at time t=1
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})  # Overlaps with node2
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.1, "y": 1.1})  # Overlaps with node1
    node3 = graph.add_node({DEFAULT_ATTR_KEYS.T: 2, "x": 2.0, "y": 2.0})

    # Add overlap - mutually exclusive pair
    graph.add_overlap(node1, node2)

    # Add edges with different weights
    # node0 -> node1 (better weight)
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})
    # node0 -> node2 (worse weight)
    edge2 = graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})
    # node1 -> node3
    edge3 = graph.add_edge(node1, node3, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})
    # node2 -> node3
    edge4 = graph.add_edge(node2, node3, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})

    solver = NearestNeighborsSolver(max_children=2)
    solver.solve(graph)

    # Check that solution respects overlap constraints
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_node_ids = selected_nodes[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

    # Verify overlap constraints are respected
    # node1 and node2 should not both be selected
    assert not (node1 in selected_node_ids and node2 in selected_node_ids)

    # Verify that the better edge (node0 -> node1) is selected
    # since node1 has better weight than node2
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_edge_ids = selected_edges[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

    # The solver should prefer the better edge (edge1: node0 -> node1)
    # and reject the worse edge (edge2: node0 -> node2) due to overlap constraint
    assert edge1 in selected_edge_ids
    assert edge2 not in selected_edge_ids
    assert edge3 in selected_edge_ids
    assert edge4 not in selected_edge_ids


def test_nearest_neighbors_solver_solve_large_graph() -> None:
    """Test solving with a larger graph to verify algorithm correctness."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", dtype=pl.Float64)
    graph.add_node_attr_key("y", dtype=pl.Float64)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, dtype=pl.Float64)

    # Create a more complex graph structure
    # Time 0: nodes 0, 1
    # Time 1: nodes 2, 3, 4
    # Time 2: nodes 5, 6

    nodes = []
    for i in range(7):
        t = i // 3 if i < 6 else 2
        node_id = graph.add_node({DEFAULT_ATTR_KEYS.T: t, "x": float(i), "y": 0.0})
        nodes.append(node_id)

    # Add edges with various weights
    edges = [
        (0, 2, 1.0),
        (0, 3, 2.0),
        (0, 4, 3.0),  # From node 0
        (1, 2, 2.5),
        (1, 3, 1.5),
        (1, 4, 2.8),  # From node 1
        (2, 5, 1.2),
        (2, 6, 1.8),  # From node 2
        (3, 5, 1.1),
        (3, 6, 1.9),  # From node 3
        (4, 5, 2.0),
        (4, 6, 1.0),  # From node 4
    ]

    for source_idx, target_idx, weight in edges:
        graph.add_edge(nodes[source_idx], nodes[target_idx], {DEFAULT_ATTR_KEYS.EDGE_DIST: weight})

    solver = NearestNeighborsSolver(max_children=2)
    solver.solve(graph)

    # Verify that solution is found
    edge_attrs = graph.edge_attrs()
    node_attrs = graph.node_attrs()

    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    # Should have some selected edges and nodes
    assert len(selected_edges) > 0
    assert len(selected_nodes) > 0

    # Verify max_children constraint
    for source_idx in [0, 1, 2, 3, 4]:
        source_edges = selected_edges.filter(selected_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE] == nodes[source_idx])
        assert len(source_edges) <= 2  # max_children constraint

    # Verify one parent constraint
    for target_idx in [2, 3, 4, 5, 6]:
        target_edges = selected_edges.filter(selected_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET] == nodes[target_idx])
        assert len(target_edges) <= 1  # one parent constraint
