import re
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy

import polars as pl
import pytest

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph, GraphView, SQLGraph
from tracksdata.utils._logging import LOG


def parametrize_subgraph_tests(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to parametrize tests for both original graphs and subgraphs."""
    return pytest.mark.parametrize("use_subgraph", [False, True], ids=["original", "subgraph"])(func)


def create_test_graph(graph_backend: BaseGraph, use_subgraph: bool = False) -> BaseGraph:
    """
    Helper function to create a test graph with multiple nodes and edges.

    Parameters
    ----------
    graph_backend : BaseGraph
        The graph backend to use for creating the test graph.
    use_subgraph : bool
        If True, returns a subgraph; if False, returns the original graph.

    Returns
    -------
    BaseGraph
        Either the original graph or a subgraph with test data.
    """
    # Add attribute keys
    graph_backend.add_node_attr_key("x", -1.0)
    graph_backend.add_node_attr_key("y", -1.0)
    graph_backend.add_node_attr_key("label", "0")
    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge_attr_key("new_attribute", 0.0)

    # Add nodes with various attributes
    node0 = graph_backend.add_node({"t": 0, "x": 0.0, "y": 0.0, "label": "0"})
    node1 = graph_backend.add_node({"t": 1, "x": 1.0, "y": 2.0, "label": "A"})
    node2 = graph_backend.add_node({"t": 2, "x": 2.0, "y": 3.0, "label": "B"})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0, "y": 4.0, "label": "A"})
    node4 = graph_backend.add_node({"t": 3, "x": 4.0, "y": 5.0, "label": "C"})
    node5 = graph_backend.add_node({"t": 3, "x": 5.0, "y": 6.0, "label": "A"})

    # Add edges only between adjacent time points (t -> t+1)
    edge0 = graph_backend.add_edge(node0, node1, attrs={"weight": 0.5, "new_attribute": 1.0})  # t=0 -> t=1
    edge1 = graph_backend.add_edge(node1, node2, attrs={"weight": 0.5, "new_attribute": 1.0})  # t=1 -> t=2
    edge2 = graph_backend.add_edge(node1, node3, attrs={"weight": 0.7, "new_attribute": 2.0})  # t=1 -> t=2
    edge3 = graph_backend.add_edge(node2, node4, attrs={"weight": 0.3, "new_attribute": 3.0})  # t=2 -> t=3
    edge4 = graph_backend.add_edge(node3, node5, attrs={"weight": 0.9, "new_attribute": 4.0})  # t=2 -> t=3

    if use_subgraph:
        # Create subgraph with nodes 1, 2, 4 in UNSORTED order to test robustness
        subgraph_nodes = [node4, node1, node2]  # Intentionally unsorted order
        subgraph = graph_backend.filter(node_ids=subgraph_nodes).subgraph()

        # Store the subgraph nodes for testing (keep original order for assertions)
        subgraph._test_nodes = [node2, node4, node1]  # type: ignore
        subgraph._test_edges = [edge3, edge1]  # edges within the subgraph (both go from t=0 to t=1)  # type: ignore
        subgraph._is_subgraph = True  # type: ignore
        return subgraph
    else:
        # Store original nodes and edges for reference
        graph_backend._test_nodes = [node0, node1, node2, node3, node4, node5]  # type: ignore
        graph_backend._test_edges = [edge0, edge1, edge2, edge3, edge4]  # type: ignore
        graph_backend._is_subgraph = False  # type: ignore
        return graph_backend


@parametrize_subgraph_tests
def test_node_ids_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test retrieving node IDs on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    node_ids = graph_with_data.node_ids()
    expected_nodes = graph_with_data._test_nodes  # type: ignore

    assert set(node_ids) == set(expected_nodes)
    assert len(node_ids) == len(expected_nodes)


@parametrize_subgraph_tests
def test_filter_nodes_and_edges_by_attr_with_data(
    graph_backend: BaseGraph,
    use_subgraph: bool,
) -> None:
    """Test filtering nodes and edges by attributes on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    node_attrs = graph_with_data.node_attrs()

    # Filter by time
    nodes = graph_with_data.filter(NodeAttr("t") == 2).node_ids()
    # Should find nodes with t=2 that are in this graph
    expected_t0_nodes = node_attrs.filter(pl.col("t") == 2)[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    assert set(nodes) == set(expected_t0_nodes)

    # Filter by label
    nodes = graph_with_data.filter(NodeAttr("label") == "A").node_ids()
    # Should find nodes with label="A" that are in this graph
    expected_label_a_nodes = node_attrs.filter(pl.col("label") == "A")[DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    assert set(nodes) == set(expected_label_a_nodes)

    edge_attrs = graph_with_data.edge_attrs()

    # including targets
    nodes = graph_with_data.filter(NodeAttr("label") == "A", include_targets=True).node_ids()
    expected_label_a_nodes.extend(
        edge_attrs.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_SOURCE) == expected_label_a_nodes[0])[
            DEFAULT_ATTR_KEYS.EDGE_TARGET
        ].to_list()
    )
    assert set(nodes) == set(expected_label_a_nodes)

    # Filter by edge weight
    edge_filter = graph_with_data.filter(EdgeAttr("weight") >= 0.4)

    # testing edges
    subgraph_edge_attrs = edge_filter.edge_attrs()
    # Should find edges with weight=0.4 that are in this graph
    expected_weight_0_4_edges = edge_attrs.filter(pl.col("weight") >= 0.4)[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()  # type: ignore
    assert set(subgraph_edge_attrs[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()) == set(expected_weight_0_4_edges)

    # testing edge_attrs
    nodes = edge_filter.node_ids()
    expected_nodes = (
        edge_attrs.filter(pl.col("weight") >= 0.4)
        .select(DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .to_numpy()
        .ravel()
        .tolist()
    )
    assert set(nodes) == set(expected_nodes)

    edge_tuples = set(subgraph_edge_attrs.iter_rows())
    expected_edge_tuples = set(
        edge_attrs.filter(pl.col("weight") >= 0.4).select(subgraph_edge_attrs.columns).iter_rows()
    )
    assert edge_tuples == expected_edge_tuples


@parametrize_subgraph_tests
def test_time_points_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test retrieving time points on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    time_points = graph_with_data.time_points()

    # Get unique time points from the nodes in this graph
    expected_times = set()
    for node in graph_with_data._test_nodes:  # type: ignore
        node_data = graph_with_data.filter(node_ids=[node]).node_attrs()
        expected_times.add(node_data["t"].to_list()[0])

    assert set(time_points) == expected_times


@parametrize_subgraph_tests
def test_node_attrs_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test retrieving node attributeson both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    nodes = graph_with_data._test_nodes[:2]  # Test with first two nodes  # type: ignore

    df = graph_with_data.filter(node_ids=nodes).node_attrs(attr_keys=["x", "y"])
    assert isinstance(df, pl.DataFrame)
    assert len(df) == len(nodes)
    assert "x" in df.columns
    assert "y" in df.columns

    # Test with single attribute key as string
    df_single = graph_with_data.filter(node_ids=nodes).node_attrs(attr_keys=["x"])
    assert "x" in df_single.columns
    assert len(df_single) == len(nodes)


@parametrize_subgraph_tests
def test_edge_attrs_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test retrieving edge attributeson both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    df = graph_with_data.edge_attrs(attr_keys=["weight"])
    assert isinstance(df, pl.DataFrame)

    assert len(df) == len(graph_with_data._test_edges)  # type: ignore

    assert "weight" in df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in df.columns


@parametrize_subgraph_tests
def test_add_node_attr_key_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test adding new node attribute keys on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add a new attribute key with default value
    graph_with_data.add_node_attr_key("new_node_attribute", 42)

    # Check that all nodes have this attribute with the default value
    nodes = graph_with_data._test_nodes  # type: ignore
    df = graph_with_data.filter(node_ids=nodes).node_attrs(attr_keys=["new_node_attribute"])

    assert all(val == 42 for val in df["new_node_attribute"].to_list())


@parametrize_subgraph_tests
def test_add_edge_attr_key_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test adding new edge attribute keys on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add a new edge attribute key with default value
    graph_with_data.add_edge_attr_key("new_edge_attribute", 99)

    # Check that all edges have this attribute with the default value
    df = graph_with_data.edge_attrs(attr_keys=["new_edge_attribute"])

    if len(df) > 0:
        assert all(val == 99 for val in df["new_edge_attribute"].to_list())


@parametrize_subgraph_tests
def test_update_node_attrs_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test updating node attributeson both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    nodes = graph_with_data._test_nodes[:2]  # Use first two nodes  # type: ignore

    # Update with single value
    graph_with_data.update_node_attrs(node_ids=[nodes[0]], attrs={"x": 99.0})

    df = graph_with_data.filter(node_ids=[nodes[0]]).node_attrs(attr_keys=["x"])
    assert df["x"].to_list()[0] == 99.0

    # Update with list of values
    graph_with_data.update_node_attrs(node_ids=nodes, attrs={"x": [100.0, 101.0]})

    df = graph_with_data.filter(node_ids=nodes).node_attrs(attr_keys=["x"])
    assert df["x"].to_list() == [100.0, 101.0]

    # test updating all nodes
    graph_with_data.update_node_attrs(attrs={"x": 0.0})

    df = graph_with_data.node_attrs(attr_keys=["x"])
    assert (df["x"] == 0.0).all()

    # Test error with wrong length
    with pytest.raises(ValueError):
        graph_with_data.update_node_attrs(node_ids=nodes, attrs={"x": [1.0]})


@parametrize_subgraph_tests
def test_update_edge_attrs_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test updating edge attributeson both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    df = graph_with_data.edge_attrs()

    # Get the first edge ID
    edge_id = df[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()[0]

    # Update edge attribute
    graph_with_data.update_edge_attrs(edge_ids=[edge_id], attrs={"weight": 99.0})

    # Verify update
    df_updated = graph_with_data.edge_attrs(attr_keys=["weight"])
    edge_weights = df_updated["weight"].to_list()
    assert 99.0 in edge_weights

    # test updating all edges
    graph_with_data.update_edge_attrs(attrs={"weight": 0.0})
    df = graph_with_data.edge_attrs(attr_keys=["weight"])
    assert (df["weight"] == 0.0).all()

    # Test error with wrong length
    with pytest.raises(ValueError):
        graph_with_data.update_edge_attrs(edge_ids=[edge_id], attrs={"weight": [1.0, 2.0]})


@parametrize_subgraph_tests
def test_num_edges_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test counting edges on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    num_edges = graph_with_data.num_edges
    assert num_edges == len(graph_with_data._test_edges)  # type: ignore


@parametrize_subgraph_tests
def test_num_nodes_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test counting nodes on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    num_nodes = graph_with_data.num_nodes
    assert num_nodes == len(graph_with_data._test_nodes)  # type: ignore


def test_subgraph_creation(graph_backend: BaseGraph) -> None:
    """Test creating subgraphs and their properties."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Create a subgraph with subset of nodes in UNSORTED order
    original_nodes = graph_with_data._test_nodes  # type: ignore
    subgraph_nodes_unsorted = [original_nodes[1], original_nodes[0]]  # Reverse order

    subgraph = graph_with_data.filter(node_ids=subgraph_nodes_unsorted).subgraph()

    # Test that subgraph has correct number of nodes
    assert subgraph.num_nodes == len(subgraph_nodes_unsorted)

    # Test that subgraph node IDs match expected (regardless of input order)
    subgraph_node_ids = subgraph.node_ids()
    assert set(subgraph_node_ids) == set(subgraph_nodes_unsorted)

    # Test that subgraph has correct node attributes
    for node in subgraph_nodes_unsorted:
        original_attrs = graph_with_data.filter(node_ids=[node]).node_attrs()
        subgraph_attrs = subgraph.filter(node_ids=[node]).node_attrs()

        # Compare all columns
        for col in original_attrs.columns:
            assert original_attrs[col].to_list() == subgraph_attrs[col].to_list()


def test_subgraph_edge_preservation(graph_backend: BaseGraph) -> None:
    """Test that subgraphs preserve correct edges between included nodes."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Get all nodes and create subgraph with first 2 nodes in REVERSE order
    original_nodes = graph_with_data._test_nodes  # type: ignore
    subgraph_nodes = [original_nodes[1], original_nodes[0]]  # Reverse order

    # Get edges from original graph involving only these nodes
    original_edges = graph_with_data.edge_attrs()
    subgraph_node_set = set(subgraph_nodes)

    expected_edges_count = 0
    for i in range(len(original_edges)):
        source = original_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        target = original_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        if source in subgraph_node_set and target in subgraph_node_set:
            expected_edges_count += 1

    # Create subgraph
    subgraph = graph_with_data.filter(node_ids=subgraph_nodes).subgraph()
    subgraph_edges = subgraph.edge_attrs()

    # Check that subgraph has the correct number of edges
    assert len(subgraph_edges) == expected_edges_count

    # Check that all edges in subgraph involve only subgraph nodes
    for i in range(len(subgraph_edges)):
        source = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        target = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        assert source in subgraph_node_set
        assert target in subgraph_node_set


def test_subgraph_attr_consistency(graph_backend: BaseGraph) -> None:
    """Test that subgraph node and edge attributesare consistent with original graph."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore
    # Use nodes in unsorted order to test robustness
    subgraph_nodes = [original_nodes[1], original_nodes[0]]

    # Create subgraph
    subgraph = graph_with_data.filter(node_ids=subgraph_nodes).subgraph()

    # Test node attribute consistency
    for node in subgraph_nodes:
        original_node_attrs = graph_with_data.filter(node_ids=[node]).node_attrs()
        subgraph_node_attrs = subgraph.filter(node_ids=[node]).node_attrs()

        # All columns should match
        assert set(original_node_attrs.columns) == set(subgraph_node_attrs.columns)

        # All values should match
        for col in original_node_attrs.columns:
            assert original_node_attrs[col].to_list() == subgraph_node_attrs[col].to_list()

    # Test edge attribute consistency for edges that exist in both
    original_edges = graph_with_data.edge_attrs()
    subgraph_edges = subgraph.edge_attrs()

    # Check that subgraph edges have same attribute keys as original
    assert set(original_edges.columns) == set(subgraph_edges.columns)

    # For each edge in subgraph, verify it has same weights as in original
    for i in range(len(subgraph_edges)):
        sub_source = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        sub_target = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        sub_weight = subgraph_edges["weight"].to_list()[i]

        # Find corresponding edge in original graph
        found_match = False
        for j in range(len(original_edges)):
            orig_source = original_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[j]
            orig_target = original_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[j]
            orig_weight = original_edges["weight"].to_list()[j]

            if orig_source == sub_source and orig_target == sub_target:
                assert orig_weight == sub_weight
                found_match = True
                break

        assert found_match, f"Edge ({sub_source}, {sub_target}) not found in original graph"


def test_subgraph_with_unsorted_node_ids(graph_backend: BaseGraph) -> None:
    """Test that subgraph creation works correctly with unsorted node IDs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Test with various unsorted orders
    test_cases = [
        [original_nodes[2], original_nodes[0], original_nodes[1]],  # Mixed order
        [original_nodes[3], original_nodes[1], original_nodes[5], original_nodes[0]],  # Descending + mixed
        [original_nodes[4], original_nodes[2]],  # Just two nodes in reverse order
    ]

    for unsorted_nodes in test_cases:
        # Create subgraph with unsorted node IDs
        subgraph = graph_with_data.filter(node_ids=unsorted_nodes).subgraph()

        # Verify the subgraph contains exactly the expected nodes
        subgraph_node_ids = set(subgraph.node_ids())
        expected_node_ids = set(unsorted_nodes)
        assert subgraph_node_ids == expected_node_ids, f"Failed for node order: {unsorted_nodes}"

        expected_edges = graph_with_data.edge_attrs()
        expected_edge_ids = expected_edges.filter(
            pl.col(DEFAULT_ATTR_KEYS.EDGE_SOURCE).is_in(unsorted_nodes)
            & pl.col(DEFAULT_ATTR_KEYS.EDGE_TARGET).is_in(unsorted_nodes)
        )[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()

        subgraph_edge_ids = set(subgraph.edge_ids())
        assert subgraph_edge_ids == set(expected_edge_ids), f"Failed for edge order: {unsorted_nodes}"

        # Verify node attributesare preserved correctly
        for node in unsorted_nodes:
            original_attrs = graph_with_data.filter(node_ids=[node]).node_attrs()
            subgraph_attrs = subgraph.filter(node_ids=[node]).node_attrs()

            for col in original_attrs.columns:
                msg = f"Node {node} attribute {col} mismatch in subgraph with order {unsorted_nodes}"
                assert original_attrs[col].to_list() == subgraph_attrs[col].to_list(), msg


def test_subgraph_add_node(graph_backend: BaseGraph) -> None:
    """Test adding nodes to a subgraph."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Create a subgraph with two nodes
    subgraph_nodes = [original_nodes[1], original_nodes[3]]
    subgraph = graph_with_data.filter(node_ids=subgraph_nodes).subgraph()

    initial_counts = (subgraph.num_nodes, graph_with_data.num_nodes)

    # Add a new node to the subgraph
    new_node_id = subgraph.add_node({"t": 10, "x": 10.0, "y": 10.0, "label": "NEW"})

    # Verify node was added to both subgraph and original graph
    assert subgraph.num_nodes == initial_counts[0] + 1
    assert graph_with_data.num_nodes == initial_counts[1] + 1
    assert new_node_id in subgraph.node_ids()
    assert new_node_id in graph_with_data.node_ids()

    # Verify attributes in both graphs
    for graph in [subgraph, graph_with_data]:
        attributes = graph.filter(node_ids=[new_node_id]).node_attrs()
        assert attributes["t"].to_list()[0] == 10
        assert attributes["label"].to_list()[0] == "NEW"


def test_subgraph_add_edge(graph_backend: BaseGraph) -> None:
    """Test adding edges to a subgraph."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Create a subgraph with three nodes
    subgraph_nodes = original_nodes[:3]
    subgraph = graph_with_data.filter(node_ids=subgraph_nodes).subgraph()

    initial_counts = (subgraph.num_edges, graph_with_data.num_edges)
    node_a, node_b = subgraph_nodes[0], subgraph_nodes[2]

    # Add a new edge between existing nodes
    subgraph.add_edge(node_a, node_b, attrs={"weight": 1.5, "new_attribute": 10.0})

    # Verify edge was added to both subgraph and original graph
    assert subgraph.num_edges == initial_counts[0] + 1
    assert graph_with_data.num_edges == initial_counts[1] + 1

    # Test both the subgraph and the original graph
    for edge_df in [subgraph.edge_attrs(), graph_with_data.edge_attrs()]:
        # Find the new edge in the edge attributes
        new_edge_found = False
        for i in range(len(edge_df)):
            source = edge_df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
            target = edge_df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
            weight = edge_df["weight"].to_list()[i]
            new_attribute = edge_df["new_attribute"].to_list()[i]

            if source == node_a and target == node_b and weight == 1.5 and new_attribute == 10.0:
                new_edge_found = True
                break

        assert new_edge_found, "New edge not found in graph edge attributes"


def test_subgraph_add_node_then_edge(graph_backend: BaseGraph) -> None:
    """Test adding a node to a subgraph and then adding an edge to/from it."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Create a subgraph with two nodes
    subgraph_nodes = [original_nodes[3], original_nodes[1]]
    subgraph = graph_with_data.filter(node_ids=subgraph_nodes).subgraph()

    # Add node and edge
    new_node_id = subgraph.add_node({"t": 20, "x": 20.0, "y": 20.0, "label": "ADDED"})
    subgraph.add_edge(subgraph_nodes[0], new_node_id, attrs={"weight": 2.0, "new_attribute": 20.0})

    # Verify propagation to both graphs
    for graph in [subgraph, graph_with_data]:
        assert new_node_id in graph.node_ids()

        edges = graph.edge_attrs()
        edge_found = any(
            edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i] == subgraph_nodes[0]
            and edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i] == new_node_id
            and edges["weight"].to_list()[i] == 2.0
            for i in range(len(edges))
        )
        assert edge_found, f"Edge to new node not found in {type(graph).__name__}"


def test_nested_subgraph_creation(graph_backend: BaseGraph) -> None:
    """Test creating a subgraph of a subgraph (nested subgraphs)."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Create nested subgraphs
    first_subgraph = graph_with_data.filter(node_ids=original_nodes[:4]).subgraph()

    second_subgraph = first_subgraph.filter(node_ids=original_nodes[:2]).subgraph()

    # Verify nested subgraph properties
    assert second_subgraph.num_nodes == 2
    assert set(second_subgraph.node_ids()) == set(original_nodes[:2])

    # Verify node attributesare preserved through all levels
    for node_id in original_nodes[:2]:
        graphs = [graph_with_data, first_subgraph, second_subgraph]
        attributes = [g.filter(node_ids=[node_id]).node_attrs() for g in graphs]

        # All should have identical attributes
        for col in attributes[0].columns:
            values = [f[col].to_list()[0] for f in attributes]
            assert all(v == values[0] for v in values), f"Attribute {col} inconsistent across graph levels"


def test_nested_subgraph_edge_preservation(graph_backend: BaseGraph) -> None:
    """Test that nested subgraphs preserve correct edges."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Create nested subgraphs: original -> first (3 nodes) -> nested (2 nodes)
    first_subgraph = graph_with_data.filter(node_ids=original_nodes[:3]).subgraph()
    nested_subgraph = first_subgraph.filter(node_ids=original_nodes[:2]).subgraph()

    # Verify the nested subgraph has exactly one edge: 0->1
    nested_edges = nested_subgraph.edge_attrs()
    assert len(nested_edges) == 1

    edge_source = nested_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[0]
    edge_target = nested_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[0]

    assert (edge_source, edge_target) == (original_nodes[0], original_nodes[1])


def test_nested_subgraph_modifications(graph_backend: BaseGraph) -> None:
    """Test adding nodes and edges to nested subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Create nested subgraphs
    first_subgraph = graph_with_data.filter(node_ids=original_nodes[:3]).subgraph()
    nested_subgraph = first_subgraph.filter(node_ids=original_nodes[:2]).subgraph()

    initial_counts = (nested_subgraph.num_nodes, graph_with_data.num_nodes)

    # Add node and edge to nested subgraph
    new_node_id = nested_subgraph.add_node({"t": 30, "x": 30.0, "y": 30.0, "label": "NESTED"})
    nested_subgraph.add_edge(original_nodes[0], new_node_id, attrs={"weight": 3.0, "new_attribute": 30.0})

    # Verify propagation to all graph levels
    # intermediate graph (first_subgraph) is not affected
    graphs = [nested_subgraph, graph_with_data]
    expected_counts = [c + 1 for c in initial_counts]

    for i, graph in enumerate(graphs):
        assert graph.num_nodes == expected_counts[i]
        assert new_node_id in graph.node_ids()

        # Verify edge exists in all graphs
        edges = graph.edge_attrs()
        edge_found = any(
            edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[j] == original_nodes[0]
            and edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[j] == new_node_id
            and edges["weight"].to_list()[j] == 3.0
            for j in range(len(edges))
        )
        assert edge_found, f"Edge not found in {type(graph).__name__}"


def test_subgraph_node_attr_filter(graph_backend: BaseGraph) -> None:
    """Test creating subgraphs using node attribute filters."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Test filtering by time point
    subgraph_t2 = graph_with_data.filter(NodeAttr("t") == 2).subgraph()

    # Should contain nodes with t=2 (nodes 2 and 3 from create_test_graph)
    expected_nodes = graph_with_data.filter(NodeAttr("t") == 2).node_ids()
    assert set(subgraph_t2.node_ids()) == set(expected_nodes)
    assert subgraph_t2.num_nodes == len(expected_nodes)

    # Verify node attributes are preserved
    for node_id in expected_nodes:
        original_attrs = graph_with_data.filter(node_ids=[node_id]).node_attrs()
        subgraph_attrs = subgraph_t2.filter(node_ids=[node_id]).node_attrs()

        for col in original_attrs.columns:
            assert original_attrs[col].to_list() == subgraph_attrs[col].to_list()

    # Test filtering by label
    subgraph_label_a = graph_with_data.filter(NodeAttr("label") == "A").subgraph()
    expected_label_a_nodes = graph_with_data.filter(NodeAttr("label") == "A").node_ids()
    assert set(subgraph_label_a.node_ids()) == set(expected_label_a_nodes)

    # Test filtering with multiple attributes
    subgraph_multi = graph_with_data.filter(NodeAttr("t") == 2, NodeAttr("label") == "A").subgraph()
    expected_multi_nodes = graph_with_data.filter(NodeAttr("t") == 2, NodeAttr("label") == "A").node_ids()
    assert set(subgraph_multi.node_ids()) == set(expected_multi_nodes)


def test_subgraph_edge_attr_filter(graph_backend: BaseGraph) -> None:
    """Test creating subgraphs using edge attribute filters."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Test filtering by edge weight
    subgraph_weight_05 = graph_with_data.filter(EdgeAttr("weight") == 0.5).subgraph()

    # Verify the subgraph contains nodes connected by edges with weight=0.5
    edges = subgraph_weight_05.edge_attrs()

    # All edges in the subgraph should have weight=0.5
    assert all(w == 0.5 for w in edges["weight"].to_list())

    # Get unique nodes from these edges
    sources = set(edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list())
    targets = set(edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list())
    expected_nodes = sources | targets

    assert set(subgraph_weight_05.node_ids()) == expected_nodes

    # Test filtering by new_attribute
    subgraph_attr_2 = graph_with_data.filter(EdgeAttr("new_attribute") == 2.0).subgraph()
    edges_attr_2 = subgraph_attr_2.edge_attrs()

    # All edges should have new_attribute=2.0
    assert all(f == 2.0 for f in edges_attr_2["new_attribute"].to_list())

    # Test filtering with multiple edge attributes
    subgraph_multi_edge = graph_with_data.filter(EdgeAttr("weight") == 0.7, EdgeAttr("new_attribute") == 2.0).subgraph()
    edges_multi = subgraph_multi_edge.edge_attrs()

    # All edges should match both criteria
    for i in range(len(edges_multi)):
        assert edges_multi["weight"].to_list()[i] == 0.7
        assert edges_multi["new_attribute"].to_list()[i] == 2.0


def test_subgraph_attr_filter_edge_preservation(graph_backend: BaseGraph) -> None:
    """Test that edge filtering properly preserves only edges matching the filter."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Create subgraph with edge filter that should include specific edges
    subgraph = graph_with_data.filter(EdgeAttr("weight") == 0.9).subgraph()

    # Get original edges matching the filter
    original_edges = graph_with_data.edge_attrs()
    expected_edges = []
    expected_nodes = set()

    for i in range(len(original_edges)):
        if original_edges["weight"].to_list()[i] == 0.9:
            expected_edges.append(i)
            expected_nodes.add(original_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i])
            expected_nodes.add(original_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i])

    # Verify subgraph has correct nodes and edges
    assert set(subgraph.node_ids()) == expected_nodes

    subgraph_edges = subgraph.edge_attrs()
    assert len(subgraph_edges) == len(expected_edges)

    # All edges in subgraph should have weight=0.9
    assert all(w == 0.9 for w in subgraph_edges["weight"].to_list())


def test_subgraph_attr_filter_empty_results(graph_backend: BaseGraph) -> None:
    """Test subgraph creation with filters that return no results."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Test node filter with non-existent value
    empty_subgraph_node = graph_with_data.filter(NodeAttr("t") == 999).subgraph()
    assert empty_subgraph_node.num_nodes == 0
    assert empty_subgraph_node.num_edges == 0

    # Test edge filter with non-existent value
    empty_subgraph_edge = graph_with_data.filter(EdgeAttr("weight") == 999.0).subgraph()
    assert empty_subgraph_edge.num_nodes == 0
    assert empty_subgraph_edge.num_edges == 0

    # Test node filter with impossible combination
    empty_subgraph_multi = graph_with_data.filter(NodeAttr("t") == 0, NodeAttr("label") == "NONEXISTENT").subgraph()
    assert empty_subgraph_multi.num_nodes == 0
    assert empty_subgraph_multi.num_edges == 0


def test_homemorphism(graph_backend: BaseGraph) -> None:
    """Test that filtering with no arguments returns the same graph."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    same_graph = graph_with_data.filter().subgraph()

    assert same_graph.node_ids() == graph_with_data.node_ids()
    assert same_graph.edge_ids() == graph_with_data.edge_ids()


@parametrize_subgraph_tests
def test_subgraph_overlaps_basic(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test basic overlap functionality in subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    nodes = graph_with_data._test_nodes  # type: ignore

    # Original graph has 6 nodes: [node0, node1, node2, node3, node4, node5]
    graph_with_data.add_overlap(nodes[0], nodes[1])
    graph_with_data.add_overlap(nodes[1], nodes[2])

    # Test basic overlap functionality
    assert graph_with_data.has_overlaps()
    overlaps = graph_with_data.overlaps()
    assert len(overlaps) == 2
    assert [nodes[0], nodes[1]] in overlaps
    assert [nodes[1], nodes[2]] in overlaps

    # Test filtering by specific nodes
    filtered_overlaps = graph_with_data.overlaps([nodes[1], nodes[2]])
    assert len(filtered_overlaps) == 1
    assert [nodes[1], nodes[2]] == filtered_overlaps[0]


def test_subgraph_overlaps_propagation(graph_backend: BaseGraph) -> None:
    """Test that overlaps propagate correctly between original graph and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Add overlaps to original graph
    graph_with_data.add_overlap(original_nodes[0], original_nodes[1])
    graph_with_data.add_overlap(original_nodes[2], original_nodes[3])

    # Create subgraph with subset of nodes
    subgraph_nodes = [original_nodes[0], original_nodes[1], original_nodes[2]]
    subgraph = graph_with_data.filter(node_ids=subgraph_nodes).subgraph()

    # Test that overlaps are accessible from subgraph
    assert subgraph.has_overlaps()
    subgraph_overlaps = subgraph.overlaps()

    # Should only see overlaps involving nodes in the subgraph
    expected_overlaps = [[original_nodes[0], original_nodes[1]]]
    assert subgraph_overlaps == expected_overlaps

    # Test that adding overlaps to subgraph affects original graph
    subgraph.add_overlap(original_nodes[0], original_nodes[2])

    # Check both graphs
    for graph in [subgraph, graph_with_data]:
        overlaps = graph.overlaps()
        assert [original_nodes[0], original_nodes[2]] in overlaps


@parametrize_subgraph_tests
def test_sucessors_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test getting successors of nodes on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    node_ids = graph_with_data._test_nodes

    edges_df = graph_with_data.edge_attrs()

    successors_dict = graph_with_data.successors(node_ids)
    out_degree = graph_with_data.out_degree(node_ids)

    for i, node_id in enumerate(node_ids):
        sucessors = successors_dict[node_id]
        expected_sucessors = edges_df.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_SOURCE) == node_id)[
            DEFAULT_ATTR_KEYS.EDGE_TARGET
        ].to_list()
        assert set(sucessors[DEFAULT_ATTR_KEYS.NODE_ID].to_list()) == set(expected_sucessors)
        assert out_degree[i] == len(expected_sucessors)

    # test out of sync
    if isinstance(graph_with_data, GraphView):
        graph_with_data.sync = False
        # sanity check, CI were failing because of some cleanup
        assert graph_with_data._root is not None
        graph_with_data.add_node({"t": 0, "x": 0.0, "y": 0.0, "label": "test"})
        with pytest.raises(RuntimeError, match="Out of sync graph view cannot be used to get sucessors"):
            graph_with_data.successors(node_ids)


@parametrize_subgraph_tests
def test_predecessors_with_data(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test getting predecessors of nodes on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    node_ids = graph_with_data._test_nodes
    edges_df = graph_with_data.edge_attrs()

    predecessors_dict = graph_with_data.predecessors(node_ids)
    in_degree = graph_with_data.in_degree(node_ids)

    for i, node_id in enumerate(node_ids):
        predecessors = predecessors_dict[node_id]
        expected_predecessors = edges_df.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_TARGET) == node_id)[
            DEFAULT_ATTR_KEYS.EDGE_SOURCE
        ].to_list()
        assert set(predecessors[DEFAULT_ATTR_KEYS.NODE_ID].to_list()) == set(expected_predecessors)
        assert in_degree[i] == len(expected_predecessors)

    # test out of sync
    if isinstance(graph_with_data, GraphView):
        graph_with_data.sync = False
        graph_with_data.add_node({"t": 0, "x": 0.0, "y": 0.0, "label": "test"})
        with pytest.raises(RuntimeError, match="Out of sync graph view cannot be used to get predecessors"):
            graph_with_data.predecessors(node_ids)


@parametrize_subgraph_tests
def test_bulk_add_nodes_returned_ids(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test that bulk_add_nodes returns correct node IDs on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add attribute keys for the new nodes
    graph_with_data.add_node_attr_key("z", 0.0)

    # Test bulk adding nodes
    nodes_to_add = [
        {"t": 10, "x": 10.0, "y": 10.0, "z": 1.0, "label": "bulk1"},
        {"t": 11, "x": 11.0, "y": 11.0, "z": 2.0, "label": "bulk2"},
        {"t": 12, "x": 12.0, "y": 12.0, "z": 3.0, "label": "bulk3"},
    ]

    initial_node_count = graph_with_data.num_nodes
    returned_ids = graph_with_data.bulk_add_nodes(nodes_to_add)

    # Test return type and length
    assert isinstance(returned_ids, list)
    assert len(returned_ids) == len(nodes_to_add)
    assert all(isinstance(node_id, int) for node_id in returned_ids)

    # Test uniqueness of returned IDs
    assert len(set(returned_ids)) == len(returned_ids)

    # Test that node count increased correctly
    assert graph_with_data.num_nodes == initial_node_count + len(nodes_to_add)

    # Test that all returned IDs correspond to actual nodes in the graph
    actual_node_ids = set(graph_with_data.node_ids())
    for node_id in returned_ids:
        assert node_id in actual_node_ids

    # Test that nodes have correct attributes
    for i, node_id in enumerate(returned_ids):
        node_attrs = graph_with_data.filter(node_ids=[node_id]).node_attrs()
        expected_attrs = nodes_to_add[i]

        assert node_attrs["t"].to_list()[0] == expected_attrs["t"]
        assert node_attrs["x"].to_list()[0] == expected_attrs["x"]
        assert node_attrs["y"].to_list()[0] == expected_attrs["y"]
        assert node_attrs["z"].to_list()[0] == expected_attrs["z"]
        assert node_attrs["label"].to_list()[0] == expected_attrs["label"]

    # Test empty input
    empty_result = graph_with_data.bulk_add_nodes([])
    assert empty_result == []
    assert graph_with_data.num_nodes == initial_node_count + len(nodes_to_add)  # No change


@parametrize_subgraph_tests
def test_bulk_add_edges_returned_ids(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test that bulk_add_edges returns correct edge IDs on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add attribute keys for the new edges
    graph_with_data.add_edge_attr_key("strength", 0.0)

    # Get some existing nodes to create edges between
    existing_nodes = graph_with_data._test_nodes  # type: ignore

    # Test bulk adding edges
    edges_to_add = [
        {
            "source_id": existing_nodes[0],
            "target_id": existing_nodes[1],
            "weight": 0.8,
            "strength": 1.5,
            "new_attribute": 10.0,
        },
        {
            "source_id": existing_nodes[1],
            "target_id": existing_nodes[2],  # if len(existing_nodes) > 2 else existing_nodes[0],
            "weight": 0.9,
            "strength": 2.0,
            "new_attribute": 20.0,
        },
    ]

    initial_edge_count = graph_with_data.num_edges
    returned_ids = graph_with_data.bulk_add_edges(deepcopy(edges_to_add), return_ids=True)

    # Test return type and length
    assert isinstance(returned_ids, list)
    assert len(returned_ids) == len(edges_to_add)
    assert all(isinstance(edge_id, int) for edge_id in returned_ids)

    # Test uniqueness of returned IDs
    assert len(set(returned_ids)) == len(returned_ids)

    # Test that edge count increased correctly
    assert graph_with_data.num_edges == initial_edge_count + len(edges_to_add)

    edge_attrs = graph_with_data.edge_attrs()
    edge_attr_0 = edge_attrs.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_ID) == returned_ids[0]).rows(named=True)[0]
    del edge_attr_0[DEFAULT_ATTR_KEYS.EDGE_ID]
    assert edge_attr_0 == edges_to_add[0]
    edge_attr_1 = edge_attrs.filter(pl.col(DEFAULT_ATTR_KEYS.EDGE_ID) == returned_ids[1]).rows(named=True)[0]
    del edge_attr_1[DEFAULT_ATTR_KEYS.EDGE_ID]
    assert edge_attr_1 == edges_to_add[1]

    # Test that returned IDs are valid integers (main requirement)
    # We validate that the IDs are non-negative integers
    for edge_id in returned_ids:
        assert isinstance(edge_id, int)
        assert edge_id >= 0

    # Test empty input
    empty_result = graph_with_data.bulk_add_edges([], return_ids=True)
    assert empty_result == []
    assert graph_with_data.num_edges == initial_edge_count + len(edges_to_add)  # No change


@parametrize_subgraph_tests
def test_custom_indices_subgraph(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test custom node indices functionality in subgraphs."""

    if not graph_backend.supports_custom_indices:
        pytest.skip("Graph does not support custom indices")

    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Test add_node with custom index
    custom_node_id = graph_with_data.add_node({"t": 0, "x": 10.0, "y": 20.0, "label": "custom_node"}, index=12345)
    assert custom_node_id == 12345

    # Test bulk_add_nodes with custom indices
    nodes = [
        {"t": 1, "x": 30.0, "y": 40.0, "label": "bulk_node_1"},
        {"t": 1, "x": 35.0, "y": 45.0, "label": "bulk_node_2"},
    ]
    custom_indices = [50000, 60000]

    returned_indices = graph_with_data.bulk_add_nodes(nodes, indices=custom_indices)
    assert returned_indices == custom_indices

    # Test that custom indices work with queries in subgraph
    for graph in [graph_with_data, graph_backend]:
        custom_node_df = graph.filter(node_ids=[12345]).node_attrs(attr_keys=["x", "y"])
        assert len(custom_node_df) == 1
        assert custom_node_df["x"].to_list()[0] == 10.0
        assert custom_node_df["y"].to_list()[0] == 20.0


@parametrize_subgraph_tests
def test_remove_node_basic(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test basic remove_node functionality on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    initial_nodes = graph_with_data._test_nodes.copy()  # type: ignore
    initial_node_count = graph_with_data.num_nodes

    # Remove a node that has edges
    node_to_remove = initial_nodes[1]
    graph_with_data.remove_node(node_to_remove)

    # Check node count decreased
    assert graph_with_data.num_nodes == initial_node_count - 1

    # Check that the node is no longer in the graph
    current_nodes = graph_with_data.node_ids()
    assert node_to_remove not in current_nodes

    # Check that all other expected nodes are still there
    remaining_expected_nodes = [n for n in initial_nodes if n != node_to_remove]
    for node in remaining_expected_nodes:
        assert node in current_nodes

    # Check that edges involving the removed node are gone
    current_edges = graph_with_data.edge_attrs()
    for i in range(len(current_edges)):
        source = current_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        target = current_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        assert source != node_to_remove
        assert target != node_to_remove


@parametrize_subgraph_tests
def test_remove_node_with_overlaps(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test removing nodes that have overlaps."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    nodes = graph_with_data._test_nodes  # type: ignore

    # Add some overlaps
    graph_with_data.add_overlap(nodes[0], nodes[1])
    graph_with_data.add_overlap(nodes[1], nodes[2])

    # Verify overlaps exist
    overlaps_before = graph_with_data.overlaps()
    assert len(overlaps_before) == 2

    # Remove node that's involved in overlaps
    graph_with_data.remove_node(nodes[1])

    # Check that overlaps involving removed node are gone
    overlaps_after = graph_with_data.overlaps()
    for overlap in overlaps_after:
        assert nodes[1] not in overlap


@parametrize_subgraph_tests
def test_remove_node_updates_time_points(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test that time points are properly updated when nodes are removed."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    initial_time_points = set(graph_with_data.time_points())
    nodes = graph_with_data._test_nodes  # type: ignore

    # Find a node that is the only one at its time point
    # First check what time points have only one node
    node_attrs = graph_with_data.node_attrs()
    time_counts = {}
    for row in node_attrs.iter_rows(named=True):
        t = row["t"]
        node_id = row[DEFAULT_ATTR_KEYS.NODE_ID]
        if t not in time_counts:
            time_counts[t] = []
        time_counts[t].append(node_id)

    # Find a time point with only one node
    single_node_time = None
    single_node_id = None
    for t, node_list in time_counts.items():
        if len(node_list) == 1 and node_list[0] in nodes:
            single_node_time = t
            single_node_id = node_list[0]
            break

    if single_node_time is not None:
        # Remove the single node at this time point
        graph_with_data.remove_node(single_node_id)

        # Check that the time point was removed
        final_time_points = set(graph_with_data.time_points())
        assert single_node_time not in final_time_points
        assert len(final_time_points) == len(initial_time_points) - 1


def test_graph_view_remove_edge(graph_backend: BaseGraph) -> None:
    """Ensure GraphView.remove_edge updates both the view and the root graph.

    Tests removal by endpoints and by edge_id with the view in sync mode.
    """
    # Setup root graph with attributes
    graph_backend.add_node_attr_key("x", None)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Nodes and edges
    n0 = graph_backend.add_node({"t": 0, "x": 0.0})
    n1 = graph_backend.add_node({"t": 1, "x": 1.0})
    n2 = graph_backend.add_node({"t": 2, "x": 2.0})

    graph_backend.add_edge(n0, n1, {"weight": 0.2})
    graph_backend.add_edge(n1, n2, {"weight": 0.8})

    # Create a view containing nodes t <= 1 (n0, n1) which includes edge n0->n1
    view = graph_backend.filter(NodeAttr("t") <= 1).subgraph()
    assert isinstance(view, GraphView)

    # Sanity: edge present in both root and view
    assert graph_backend.has_edge(n0, n1)
    assert view.has_edge(n0, n1)

    # Remove by endpoints in the view
    view.remove_edge(n0, n1)
    assert not graph_backend.has_edge(n0, n1)
    assert not view.has_edge(n0, n1)

    # Re-add in the view (which syncs to root) and get the edge id from the view
    view.add_edge(n0, n1, {"weight": 0.9})
    vid = view.edge_attrs()[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()[0]

    # Remove by edge_id via the view
    view.remove_edge(edge_id=vid)
    assert not view.has_edge(n0, n1)
    assert not graph_backend.has_edge(n0, n1)

    # Removing a non-existent edge via the view raises
    with pytest.raises(ValueError, match=rf"Edge {n0}->{n1} does not exist in the graph\."):
        view.remove_edge(n0, n1)
    with pytest.raises(ValueError, match=rf"Edge {vid} does not exist in the graph\."):
        view.remove_edge(edge_id=vid)
    with pytest.raises(ValueError, match=r"Provide either edge_id or both source_id and target_id\."):
        view.remove_edge()


@parametrize_subgraph_tests
def test_has_edge(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test has_edge functionality on both original graphs and subgraphs."""

    @contextmanager
    def _ignore_index_map_warnings():
        def flt(r) -> bool:
            return not re.search(r"not found in index map", r.getMessage())

        LOG.addFilter(flt)
        try:
            yield
        finally:
            LOG.removeFilter(flt)

    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add some edges
    edge_attrs = graph_with_data.edge_attrs()

    for src_id, tgt_id in zip(
        edge_attrs[DEFAULT_ATTR_KEYS.EDGE_SOURCE], edge_attrs[DEFAULT_ATTR_KEYS.EDGE_TARGET], strict=False
    ):
        assert graph_with_data.has_edge(src_id, tgt_id)

    with _ignore_index_map_warnings():
        assert not graph_with_data.has_edge(10, 15)

    assert not graph_with_data.has_edge(graph_with_data._test_nodes[0], graph_with_data._test_nodes[-1])

    # check if filtered edges are not present in the subgraph (they are in the _root graph)
    if isinstance(graph_with_data, GraphView):
        parent = graph_with_data._root
        parent_edges = parent.edge_attrs().select(["source_id", "target_id"]).to_numpy().tolist()
        graph_edges = graph_with_data.edge_attrs().select(["source_id", "target_id"]).to_numpy().tolist()

        for edge in parent_edges:
            if edge not in graph_edges:
                with _ignore_index_map_warnings():
                    assert not graph_with_data.has_edge(edge[0], edge[1])
            else:
                assert graph_with_data.has_edge(edge[0], edge[1])


@parametrize_subgraph_tests
def test_edge_id(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test edge_id functionality on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add some edges
    edge_attrs = graph_with_data.edge_attrs()

    for attr in edge_attrs.rows(named=True):
        edge_id = graph_with_data.edge_id(attr[DEFAULT_ATTR_KEYS.EDGE_SOURCE], attr[DEFAULT_ATTR_KEYS.EDGE_TARGET])
        assert edge_id == attr[DEFAULT_ATTR_KEYS.EDGE_ID]


@parametrize_subgraph_tests
def test_graph_copy(graph_backend: BaseGraph, use_subgraph: bool) -> None:
    """Test copy functionality on both original graphs and subgraphs."""

    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    if isinstance(graph_with_data, SQLGraph):
        kwargs = {
            "drivername": "sqlite",
            "database": ":memory:",
            "overwrite": True,
        }
    else:
        kwargs = {}

    if isinstance(graph_with_data, GraphView):
        with pytest.raises(ValueError):
            copied_graph = graph_with_data.copy(**kwargs)

    else:
        copied_graph = graph_with_data.copy(**kwargs)
        assert copied_graph.num_nodes == graph_with_data.num_nodes
        assert copied_graph.num_edges == graph_with_data.num_edges
        assert copied_graph.node_ids() == graph_with_data.node_ids()
        assert copied_graph.edge_ids() == graph_with_data.edge_ids()
