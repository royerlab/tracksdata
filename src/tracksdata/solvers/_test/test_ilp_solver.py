import math

import pytest

from tracksdata.attrs import Attr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.solvers import ILPSolver


def test_ilp_solver_init_default() -> None:
    """Test ILPSolver initialization with default parameters."""
    solver = ILPSolver()

    assert solver.output_key == DEFAULT_ATTR_KEYS.SOLUTION
    assert isinstance(solver.edge_weight_expr, Attr)
    assert isinstance(solver.node_weight_expr, Attr)
    assert isinstance(solver.appearance_weight_expr, Attr)
    assert isinstance(solver.disappearance_weight_expr, Attr)
    assert isinstance(solver.division_weight_expr, Attr)


def test_ilp_solver_init_custom() -> None:
    """Test ILPSolver initialization with custom parameters."""
    solver = ILPSolver(
        edge_weight="custom_edge_weight",
        node_weight="custom_node_weight",
        appearance_weight=1.5,
        disappearance_weight=-0.5,
        division_weight=2.0,
        output_key="custom_solution",
        timeout=60.0,
    )

    assert solver.output_key == "custom_solution"
    assert isinstance(solver.edge_weight_expr, Attr)
    assert isinstance(solver.node_weight_expr, Attr)
    assert isinstance(solver.appearance_weight_expr, Attr)
    assert isinstance(solver.disappearance_weight_expr, Attr)
    assert isinstance(solver.division_weight_expr, Attr)


def test_ilp_solver_init_with_attr_expr() -> None:
    """Test ILPSolver initialization with AttrExpr objects."""
    edge_weight_expr = Attr("edge_weight") * 2
    node_weight_expr = -Attr("node_weight")

    solver = ILPSolver(edge_weight=edge_weight_expr, node_weight=node_weight_expr)

    # Check that the expressions have the same underlying polars expression
    assert str(solver.edge_weight_expr) == str(edge_weight_expr)
    assert str(solver.node_weight_expr) == str(node_weight_expr)


def test_ilp_solver_reset_model() -> None:
    """Test that reset_model clears all internal state."""
    solver = ILPSolver()

    # Manually modify some state
    solver._count = 10
    solver._node_vars = {"test": "value"}

    solver.reset_model()

    assert solver._count == 0
    assert len(solver._node_vars) == 0
    assert len(solver._edge_vars) == 0
    assert len(solver._appear_vars) == 0
    assert len(solver._disappear_vars) == 0
    assert len(solver._division_vars) == 0


def test_ilp_solver_solve_empty_graph() -> None:
    """Test solving on an empty graph."""
    graph = RustWorkXGraph()
    solver = ILPSolver()

    # Should not raise an error on empty graph
    with pytest.raises(ValueError, match=r"Empty ILPSolver model, there is nothing to solve\."):
        solver.solve(graph)


def test_ilp_solver_solve_no_edges(caplog: pytest.LogCaptureFixture) -> None:
    """Test solving on a graph with nodes but no edges."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)

    # Add some nodes
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    solver = ILPSolver(
        appearance_weight=1.0,
        disappearance_weight=1.0,
        division_weight=1.0,
    )

    with pytest.raises(ValueError):
        solver.solve(graph)


def test_ilp_solver_solve_simple_case() -> None:
    """Test solving with a simple graph."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges with weights (negative weights for minimization)
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})
    graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})

    solver = ILPSolver(return_solution=True)
    solution_graph = solver.solve(graph)

    # Check that solution keys are added
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_attrs.columns

    # Check that some solution is found
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_edges) > 0
    assert len(selected_nodes) > 0

    assert solution_graph is not None
    assert solution_graph.num_nodes == len(selected_nodes)
    assert solution_graph.num_edges == len(selected_edges)


def test_ilp_solver_solve_with_appearance_weight() -> None:
    """Test solving with appearance weights."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    # Add edge
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.1})

    # Use positive appearance weight to penalize appearances
    solver = ILPSolver(appearance_weight=1.0)
    solver.solve(graph)

    # Check that solution is found
    node_attrs = graph.node_attrs()
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0

    # resetting the solution
    graph.update_node_attrs(node_ids=[node0, node1], attrs={DEFAULT_ATTR_KEYS.SOLUTION: False})
    graph.update_edge_attrs(edge_ids=[edge1], attrs={DEFAULT_ATTR_KEYS.SOLUTION: False})

    # penalization is too high, empty solution
    solver = ILPSolver(appearance_weight=2.0)
    solver.solve(graph)

    node_attrs = graph.node_attrs()
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) == 0


def test_ilp_solver_solve_with_disappearance_weight() -> None:
    """Test solving with disappearance weights."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    # Add edge
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.1})

    # Use positive disappearance weight to penalize disappearances
    solver = ILPSolver(disappearance_weight=1.0)
    solver.solve(graph)

    # Check that solution is found
    node_attrs = graph.node_attrs()
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0

    # resetting the solution
    graph.update_node_attrs(node_ids=[node0, node1], attrs={DEFAULT_ATTR_KEYS.SOLUTION: False})
    graph.update_edge_attrs(edge_ids=[edge1], attrs={DEFAULT_ATTR_KEYS.SOLUTION: False})

    # penalization is too high, empty solution
    solver = ILPSolver(disappearance_weight=2.0)
    solver.solve(graph)

    node_attrs = graph.node_attrs()
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) == 0


def test_ilp_solver_solve_with_division_weight() -> None:
    """Test solving with division weights."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes for division scenario
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges to create potential division
    edge1 = graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.0})
    edge2 = graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -2.1})

    # Use positive division weight to penalize divisions
    solver = ILPSolver(
        appearance_weight=1.0,
        division_weight=3.0,
    )
    solver.solve(graph)

    # Check that solution is found
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])
    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0
    assert len(selected_edges) > 0

    # resetting the solution
    graph.update_node_attrs(node_ids=[node0, node1, node2], attrs={DEFAULT_ATTR_KEYS.SOLUTION: False})
    graph.update_edge_attrs(edge_ids=[edge1, edge2], attrs={DEFAULT_ATTR_KEYS.SOLUTION: False})

    # penalization is too high, empty solution
    solver = ILPSolver(
        appearance_weight=2.5,
        division_weight=5.0,
    )
    solver.solve(graph)

    node_attrs = graph.node_attrs()
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) == 0


def test_ilp_solver_solve_custom_edge_weight_expr() -> None:
    """Test solving with custom edge weight expression."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key("custom_weight", 0.0)
    graph.add_edge_attr_key("confidence", 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    # Add edge with custom attributes
    graph.add_edge(node0, node1, {"custom_weight": 2.0, "confidence": 0.8})

    # Use complex expression: -custom_weight * confidence
    weight_expr = -Attr("custom_weight") * Attr("confidence")
    solver = ILPSolver(edge_weight=weight_expr)
    solver.solve(graph)

    # Check that solution keys are added
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_attrs.columns


def test_ilp_solver_solve_custom_node_weight_expr() -> None:
    """Test solving with custom node weight expression."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("quality", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes with quality attribute
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "quality": 0.9})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "quality": 0.7})

    # Add edge
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})

    # Use node quality as weight (negative to encourage high quality nodes)
    node_weight_expr = -Attr("quality")
    solver = ILPSolver(node_weight=node_weight_expr)
    solver.solve(graph)

    # Check that solution is found
    node_attrs = graph.node_attrs()
    selected_nodes = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    assert len(selected_nodes) > 0


def test_ilp_solver_solve_custom_output_key() -> None:
    """Test solving with custom output key."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes and edges
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})

    custom_key = "my_ilp_solution"
    solver = ILPSolver(output_key=custom_key)
    solver.solve(graph)

    # Check that custom key is used
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert custom_key in node_attrs.columns
    assert custom_key in edge_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION not in node_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION not in edge_attrs.columns


def test_ilp_solver_solve_with_all_weights() -> None:
    """Test solving with all weight types specified."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 2.0})

    # Add edges
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.5})
    graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})

    solver = ILPSolver(
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST,
        node_weight=-0.5,  # Encourage node selection
        appearance_weight=0.3,  # Penalize appearances
        disappearance_weight=0.2,  # Penalize disappearances
        division_weight=1.0,  # Penalize divisions
    )
    solver.solve(graph)

    # Check that solution is found
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert DEFAULT_ATTR_KEYS.SOLUTION in node_attrs.columns
    assert DEFAULT_ATTR_KEYS.SOLUTION in edge_attrs.columns


def test_ilp_solver_evaluate_expr_scalar() -> None:
    """Test _evaluate_expr method with scalar expressions."""
    import polars as pl

    solver = ILPSolver()

    # Test with scalar expression (no column dependencies)
    scalar_expr = Attr(5.0)
    df = pl.DataFrame({"dummy": [1, 2, 3]})

    result = solver._evaluate_expr(scalar_expr, df)

    assert result == [5.0, 5.0, 5.0]  # Should repeat scalar for each row


def test_ilp_solver_evaluate_expr_column() -> None:
    """Test _evaluate_expr method with column expressions."""
    import polars as pl

    solver = ILPSolver()

    # Test with column expression
    column_expr = Attr("values") * 2
    df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

    result = solver._evaluate_expr(column_expr, df)

    assert result == [2.0, 4.0, 6.0]


def test_ilp_solver_division_constraint() -> None:
    """Test that division constraint is properly enforced: node_var >= division_var."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Create a scenario where division would be tempting but should be constrained
    # Time 0: 1 parent node
    parent_node = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})

    # Time 1: 2 potential children
    child1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 0.0})
    child2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": -1.0, "y": 0.0})

    # Add edges from parent to both children with negative weights
    graph.add_edge(parent_node, child1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -5.0})
    graph.add_edge(parent_node, child2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -3.0})

    # Use negative division weight to make division attractive
    # But positive edge weights to make connections costly
    # This tests that division can overcome edge costs when beneficial
    solver = ILPSolver(
        node_weight=0.0,  # Neutral node cost
        division_weight=1.5,  # Penalize divisions, but not too much
        appearance_weight=1.0,  # Penalize appearances
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST,
    )

    solver.solve(graph)

    # Check results
    node_attrs = graph.filter(
        node_ids=[parent_node, child1, child2],
    ).node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.SOLUTION])
    edge_attrs = graph.edge_attrs()

    assert node_attrs[DEFAULT_ATTR_KEYS.SOLUTION].all()

    # Get solution status for each element
    parent_selected, child1_selected, child2_selected = node_attrs[DEFAULT_ATTR_KEYS.SOLUTION].to_list()

    child2_selected = node_attrs.filter(node_attrs[DEFAULT_ATTR_KEYS.NODE_ID] == child2)[
        DEFAULT_ATTR_KEYS.SOLUTION
    ].to_list()[0]

    selected_edges = edge_attrs.filter(edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION])

    # Key test of division constraint: node_var >= division_var
    # This means division can only happen if the node is selected

    # Count edges that are selected
    num_selected_edges = len(selected_edges)

    # The key constraint test: if edges are selected, parent must be selected
    assert num_selected_edges == 2

    # Validate flow constraints: edges can only be selected if their endpoints are selected
    for edge_row in selected_edges.iter_rows(named=True):
        source_id = edge_row[DEFAULT_ATTR_KEYS.EDGE_SOURCE]
        target_id = edge_row[DEFAULT_ATTR_KEYS.EDGE_TARGET]

        # For this test, we know the structure, so we can check directly
        if source_id == parent_node:
            assert parent_selected, "Edge from parent selected but parent not selected"
        if target_id == child1:
            assert child1_selected, "Edge to child1 selected but child1 not selected"
        if target_id == child2:
            assert child2_selected, "Edge to child2 selected but child2 not selected"


def test_ilp_solver_solve_with_inf_expr() -> None:
    """Test solving with infinity expressions."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 5.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 2.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 3.0})

    # Add edges
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.5})
    graph.add_edge(node0, node2, {DEFAULT_ATTR_KEYS.EDGE_DIST: -1.0})

    solver = ILPSolver(
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST,
        division_weight=100.0 * Attr("y")
        - math.inf * (Attr("x") == 0.0),  # Despite the penalization it's selected because of inf
    )
    solver.solve(graph)

    # Check that solution is found
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION].to_list() == [True, True]
    assert node_attrs[DEFAULT_ATTR_KEYS.SOLUTION].to_list() == [True, True, True]


def test_ilp_solver_solve_with_pos_inf_rejection() -> None:
    """Test solving with positive infinity to force rejection of variables."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 0.0})  # This node will be rejected

    # Add edge with attractive weight
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: -5.0})

    solver = ILPSolver(
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST,
        node_weight=0.1 + math.inf * (Attr("x") == 0.0),  # Reject nodes where x == 0
    )
    solver.solve(graph)

    # Check that node1 is rejected despite attractive edge
    node_attrs = graph.node_attrs()
    edge_attrs = graph.edge_attrs()

    assert edge_attrs[DEFAULT_ATTR_KEYS.SOLUTION].to_list() == [False]
    assert node_attrs[DEFAULT_ATTR_KEYS.SOLUTION].to_list() == [False, False]


def test_ilp_solver_solve_with_neg_inf_node_weight() -> None:
    """Test solving with negative infinity on node weights to force selection."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("priority", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "priority": 1.0})  # High priority
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "priority": 0.0})

    # Add edge with costly weight
    graph.add_edge(node0, node1, {DEFAULT_ATTR_KEYS.EDGE_DIST: 10.0})

    solver = ILPSolver(
        edge_weight=DEFAULT_ATTR_KEYS.EDGE_DIST,
        node_weight=-math.inf * (Attr("priority") == 1.0),  # Force high priority nodes
        appearance_weight=5.0,  # Penalize appearances
    )
    solver.solve(graph)

    # Check that high priority node is selected despite costs
    node_attrs = graph.node_attrs()

    priority_node_selected = node_attrs.filter(node_attrs["priority"] == 1.0)[DEFAULT_ATTR_KEYS.SOLUTION].to_list()[0]
    assert priority_node_selected


def test_ilp_solver_solve_with_inf_edge_weight() -> None:
    """Test solving with infinity on edge weights to force edge selection/rejection."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_edge_attr_key("confidence", 0.0)

    # Add nodes
    node0 = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0})
    node1 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0})
    node2 = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0})

    # Add edges - one high confidence, one low confidence
    graph.add_edge(node0, node1, {"confidence": 0.9})  # High confidence - should be forced
    graph.add_edge(node0, node2, {"confidence": 0.1})  # Low confidence - should be rejected

    # The pinning is inverse the confidence to test if the pinning is working
    solver = ILPSolver(
        edge_weight=-Attr("confidence") + math.inf * (Attr("confidence") > 0.5) - math.inf * (Attr("confidence") < 0.5),
    )
    solver.solve(graph)

    # Check that only high confidence edge is selected
    edge_attrs = graph.edge_attrs()
    high_conf_edge = edge_attrs.filter(edge_attrs["confidence"] > 0.5)[DEFAULT_ATTR_KEYS.SOLUTION].to_list()[0]
    low_conf_edge = edge_attrs.filter(edge_attrs["confidence"] < 0.5)[DEFAULT_ATTR_KEYS.SOLUTION].to_list()[0]

    assert not high_conf_edge
    assert low_conf_edge


def test_ilp_solver_solve_with_overlaps() -> None:
    """Test solving with overlapping nodes that should be mutually exclusive."""
    graph = RustWorkXGraph()

    # Register attribute keys
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("y", 0.0)
    graph.add_edge_attr_key(DEFAULT_ATTR_KEYS.EDGE_DIST, 0.0)

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

    solver = ILPSolver()
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
