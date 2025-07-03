from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph, SQLGraph
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.io._numpy_array import load_array
from tracksdata.nodes._mask import Mask


def test_already_existing_keys(graph_backend: BaseGraph) -> None:
    """Test that adding already existing keys raises an error."""
    graph_backend.add_node_attr_key("x", None)

    with pytest.raises(ValueError):
        graph_backend.add_node_attr_key("x", None)

    with pytest.raises(ValueError):
        # missing x
        graph_backend.add_node(attrs={"t": 0})


def testing_empty_graph(graph_backend: BaseGraph) -> None:
    """Test that the graph is empty."""
    assert graph_backend.num_nodes == 0
    assert graph_backend.num_edges == 0

    assert graph_backend.node_attrs().is_empty()
    assert graph_backend.edge_attrs().is_empty()


def test_node_validation(graph_backend: BaseGraph) -> None:
    """Test node validation."""
    # 't' key must exist by default
    graph_backend.add_node({"t": 1})

    with pytest.raises(ValueError):
        graph_backend.add_node({"t": 0, "x": 1.0})


def test_edge_validation(graph_backend: BaseGraph) -> None:
    """Test edge validation."""
    with pytest.raises(ValueError):
        graph_backend.add_edge(0, 1, {"weight": 0.5})


def test_add_node(graph_backend: BaseGraph) -> None:
    """Test adding nodes with various attributes."""

    for key in ["x", "y"]:
        graph_backend.add_node_attr_key(key, None)

    node_id = graph_backend.add_node({"t": 0, "x": 1.0, "y": 2.0})
    assert isinstance(node_id, int)

    # Check node attributes
    df = graph_backend.node_attrs(node_ids=[node_id])
    assert df["t"].to_list() == [0]
    assert df["x"].to_list() == [1.0]
    assert df["y"].to_list() == [2.0]

    # checking if it's sorted
    assert graph_backend.node_attrs(attr_keys=["t", "x", "y"]).columns == ["t", "x", "y"]
    assert graph_backend.node_attrs(attr_keys=["x", "y", "t"]).columns == ["x", "y", "t"]
    assert graph_backend.node_attrs(attr_keys=["y", "t", "x"]).columns == ["y", "t", "x"]


def test_add_edge(graph_backend: BaseGraph) -> None:
    """Test adding edges with attributes."""
    # Add node attribute key
    graph_backend.add_node_attr_key("x", None)

    # Add two nodes first
    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 1.0})

    # Add edge attribute key
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Add edge
    edge_id = graph_backend.add_edge(node1, node2, attrs={"weight": 0.5})
    assert isinstance(edge_id, int)

    # Check edge attributes
    df = graph_backend.edge_attrs()
    assert df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list() == [node1]
    assert df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list() == [node2]
    assert df["weight"].to_list() == [0.5]

    # testing adding new add attribute
    graph_backend.add_edge_attr_key("new_attribute", 0.0)
    edge_id = graph_backend.add_edge(node2, node3, attrs={"new_attribute": 1.0, "weight": 0.1})
    assert isinstance(edge_id, int)

    # testing default value was assigned correctly
    # at some point there was a bug and this was needed
    # df = graph_backend.edge_attrs(node_ids=[node1, node2, node3])
    df = graph_backend.edge_attrs()
    assert df["new_attribute"].to_list() == [0.0, 1.0]
    assert df["weight"].to_list() == [0.5, 0.1]


def test_node_ids(graph_backend: BaseGraph) -> None:
    """Test retrieving node IDs."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    assert set(graph_backend.node_ids()) == {node1, node2}


def test_filter_nodes_by_attribute(graph_backend: BaseGraph) -> None:
    """Test filtering nodes by attributes."""
    graph_backend.add_node_attr_key("label", None)

    node1 = graph_backend.add_node({"t": 0, "label": "A"})
    node2 = graph_backend.add_node({"t": 0, "label": "B"})
    node3 = graph_backend.add_node({"t": 1, "label": "A"})

    # Filter by time
    nodes = graph_backend.filter_nodes_by_attrs(NodeAttr("t") == 0)
    assert set(nodes) == {node1, node2}

    # Filter by label
    nodes = graph_backend.filter_nodes_by_attrs(NodeAttr("label") == "A")
    assert set(nodes) == {node1, node3}

    # Filter by t and label using multiple conditions
    nodes = graph_backend.filter_nodes_by_attrs(
        NodeAttr("t") == 1,
        NodeAttr("label") == "A",
    )
    assert set(nodes) == {node3}

    # Test with inequality
    nodes = graph_backend.filter_nodes_by_attrs(NodeAttr("t") > 0)
    assert set(nodes) == {node3}

    # Test with multiple conditions using *args for AND
    nodes = graph_backend.filter_nodes_by_attrs(NodeAttr("t") == 0, NodeAttr("label") == "A")
    assert set(nodes) == {node1}


def test_time_points(graph_backend: BaseGraph) -> None:
    """Test retrieving time points."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 2})
    graph_backend.add_node({"t": 1})

    assert set(graph_backend.time_points()) == {0, 1, 2}


def test_node_attrs(graph_backend: BaseGraph) -> None:
    """Test retrieving node attributes."""
    graph_backend.add_node_attr_key("x", None)
    graph_backend.add_node_attr_key("coordinates", np.array([0.0, 0.0]))

    node1 = graph_backend.add_node({"t": 0, "x": 1.0, "coordinates": np.array([10.0, 20.0])})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "coordinates": np.array([30.0, 40.0])})

    df = graph_backend.node_attrs(node_ids=[node1, node2], attr_keys=["x"])
    assert isinstance(df, pl.DataFrame)
    assert df["x"].to_list() == [1.0, 2.0]

    # Test unpack functionality
    df_unpacked = graph_backend.node_attrs(node_ids=[node1, node2], attr_keys=["coordinates"], unpack=True)
    if "coordinates_0" in df_unpacked.columns:
        assert df_unpacked["coordinates_0"].to_list() == [10.0, 30.0]
        assert df_unpacked["coordinates_1"].to_list() == [20.0, 40.0]


def test_edge_attrs(graph_backend: BaseGraph) -> None:
    """Test retrieving edge attributes."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge_attr_key("vector", np.array([0.0, 0.0]))

    graph_backend.add_edge(node1, node2, attrs={"weight": 0.5, "vector": np.array([1.0, 2.0])})

    df = graph_backend.edge_attrs(attr_keys=["weight"])
    assert isinstance(df, pl.DataFrame)
    assert df["weight"].to_list() == [0.5]

    # Test unpack functionality
    df_unpacked = graph_backend.edge_attrs(attr_keys=["vector"], unpack=True)
    if "vector_0" in df_unpacked.columns:
        assert df_unpacked["vector_0"].to_list() == [1.0]
        assert df_unpacked["vector_1"].to_list() == [2.0]


def test_edge_attrs_subgraph_edge_ids(graph_backend: BaseGraph) -> None:
    """Test that edge_attrs preserves original edge IDs when using node_ids parameter."""
    # Add edge attribute key
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create nodes
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})
    node3 = graph_backend.add_node({"t": 2})
    node4 = graph_backend.add_node({"t": 3})

    print(f"Created nodes: {node1=}, {node2=}, {node3=}, {node4=}")

    # Create edges
    edge1 = graph_backend.add_edge(node1, node2, attrs={"weight": 0.1})
    edge2 = graph_backend.add_edge(node2, node3, attrs={"weight": 0.2})
    edge3 = graph_backend.add_edge(node3, node4, attrs={"weight": 0.3})
    edge4 = graph_backend.add_edge(node1, node3, attrs={"weight": 0.4})

    print(f"Created edges: {edge1=}, {edge2=}, {edge3=}, {edge4=}")

    # Get all edge attributes(full graph)
    df_full = graph_backend.edge_attrs()
    print(f"Full graph edges: {df_full}")

    full_edge_ids = df_full[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
    full_sources = df_full[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()
    full_targets = df_full[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()

    print("Full graph edge details:")
    for eid, src, tgt in zip(full_edge_ids, full_sources, full_targets, strict=False):
        print(f"  Edge {eid}: {src} -> {tgt}")

    # Get edge attributesfor a subset of nodes [node1, node2, node3]
    # This should include:
    # - edge1: node1 -> node2
    # - edge2: node2 -> node3
    # - edge4: node1 -> node3
    # But NOT edge3: node3 -> node4 (since node4 is not in the subset)
    df_subset = graph_backend.edge_attrs(node_ids=[node1, node2, node3])
    print(f"Subset graph edges: {df_subset}")

    subset_edge_ids = df_subset[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
    subset_sources = df_subset[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()
    subset_targets = df_subset[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()

    print("Subset graph edge details:")
    for eid, src, tgt in zip(subset_edge_ids, subset_sources, subset_targets, strict=False):
        print(f"  Edge {eid}: {src} -> {tgt}")

    # The edge IDs should preserve the original edge IDs
    # and only include edges between the specified nodes
    expected_subset_edge_ids = {edge1, edge2, edge4}
    actual_subset_edge_ids = set(subset_edge_ids)

    # This will demonstrate the bug
    msg = f"Expected {expected_subset_edge_ids}, got {actual_subset_edge_ids}"
    assert actual_subset_edge_ids == expected_subset_edge_ids, msg


def test_subgraph_with_node_and_edge_attr_filters(graph_backend: BaseGraph) -> None:
    """Test subgraph with node and edge attribute filters."""
    graph_backend.add_node_attr_key("x", None)
    graph_backend.add_edge_attr_key("weight", 0.0)

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 1.0})
    node4 = graph_backend.add_node({"t": 3, "x": 3.0})
    node5 = graph_backend.add_node({"t": 4, "x": 0.5})

    graph_backend.add_edge(node1, node3, attrs={"weight": 0.8})
    edge2 = graph_backend.add_edge(node3, node5, attrs={"weight": 0.2})
    graph_backend.add_edge(node2, node4, attrs={"weight": 0.0})

    subgraph = graph_backend.subgraph(
        NodeAttr("x") <= 1.0,
        EdgeAttr("weight") < 0.5,
    )

    assert subgraph.num_nodes == 3
    assert subgraph.num_edges == 1

    subgraph_node_ids = subgraph.node_ids()
    assert set(subgraph_node_ids) == {node1, node3, node5}

    subgraph_edge_ids = subgraph.edge_ids()
    assert set(subgraph_edge_ids) == {edge2}


def test_subgraph_with_node_ids_and_filters(graph_backend: BaseGraph) -> None:
    """Test subgraph with node IDs and filters."""
    graph_backend.add_node_attr_key("x", None)
    graph_backend.add_edge_attr_key("weight", 0.0)

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 1.0})
    node4 = graph_backend.add_node({"t": 3, "x": 3.0})
    node5 = graph_backend.add_node({"t": 4, "x": 0.5})

    graph_backend.add_edge(node1, node3, attrs={"weight": 0.8})
    graph_backend.add_edge(node3, node5, attrs={"weight": 0.2})
    graph_backend.add_edge(node2, node4, attrs={"weight": 0.0})

    subgraph = graph_backend.subgraph(
        NodeAttr("x") <= 1.0,
        EdgeAttr("weight") < 0.5,
        node_ids=[node1, node3],
    )

    assert subgraph.num_nodes == 2
    assert subgraph.num_edges == 0

    subgraph_node_ids = subgraph.node_ids()
    assert set(subgraph_node_ids) == {node1, node3}

    subgraph_edge_ids = subgraph.edge_ids()
    assert len(subgraph_edge_ids) == 0


def test_add_node_attr_key(graph_backend: BaseGraph) -> None:
    """Test adding new node attribute keys."""
    node = graph_backend.add_node({"t": 0})
    graph_backend.add_node_attr_key("new_attribute", 42)

    df = graph_backend.node_attrs(node_ids=[node], attr_keys=["new_attribute"])
    assert df["new_attribute"].to_list() == [42]


def test_add_edge_attr_key(graph_backend: BaseGraph) -> None:
    """Test adding new edge attribute keys."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_attr_key("new_attribute", 42)
    graph_backend.add_edge(node1, node2, attrs={"new_attribute": 42})

    df = graph_backend.edge_attrs(attr_keys=["new_attribute"])
    assert df["new_attribute"].to_list() == [42]


def test_update_node_attrs(graph_backend: BaseGraph) -> None:
    """Test updating node attributes."""
    graph_backend.add_node_attr_key("x", None)

    node_1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node_2 = graph_backend.add_node({"t": 0, "x": 2.0})

    graph_backend.update_node_attrs(node_ids=[node_1], attrs={"x": 3.0})

    df = graph_backend.node_attrs(node_ids=[node_1, node_2], attr_keys="x")
    assert df["x"].to_list() == [3.0, 2.0]

    # inverted access on purpose
    graph_backend.update_node_attrs(node_ids=[node_2, node_1], attrs={"x": [5.0, 6.0]})

    df = graph_backend.node_attrs(node_ids=[node_1, node_2], attr_keys="x")
    assert df["x"].to_list() == [6.0, 5.0]

    # wrong length
    with pytest.raises(ValueError):
        graph_backend.update_node_attrs(node_ids=[node_1, node_2], attrs={"x": [1.0]})


def test_update_edge_attrs(graph_backend: BaseGraph) -> None:
    """Test updating edge attributes."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_attr_key("weight", 0.0)
    edge_id = graph_backend.add_edge(node1, node2, attrs={"weight": 0.5})

    graph_backend.update_edge_attrs(edge_ids=[edge_id], attrs={"weight": 1.0})
    df = graph_backend.edge_attrs(node_ids=[node1, node2], attr_keys=["weight"])
    assert df["weight"].to_list() == [1.0]

    # wrong length
    with pytest.raises(ValueError):
        graph_backend.update_edge_attrs(edge_ids=[edge_id], attrs={"weight": [1.0, 2.0]})


def test_num_edges(graph_backend: BaseGraph) -> None:
    """Test counting edges."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attrs={"weight": 0.5})

    assert graph_backend.num_edges == 1


def test_num_nodes(graph_backend: BaseGraph) -> None:
    """Test counting nodes."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 1})

    assert graph_backend.num_nodes == 2


def test_edge_attrs_include_targets(graph_backend: BaseGraph) -> None:
    """Test the inclusive flag behavior in edge_attrs method."""
    # Add edge attribute key
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create a graph with 4 nodes
    # Graph structure:
    #   node0 -> node1 -> node2 -> node3
    #        \             ^
    #         -> node3  ----+
    node0 = graph_backend.add_node({"t": 0})
    node1 = graph_backend.add_node({"t": 1})
    node2 = graph_backend.add_node({"t": 2})
    node3 = graph_backend.add_node({"t": 3})

    print(f"Created nodes: {node0=}, {node1=}, {node2=}, {node3=}")

    # Create edges with different weights for easy identification
    edge0 = graph_backend.add_edge(node0, node1, attrs={"weight": 0.1})  # node0 -> node1
    edge1 = graph_backend.add_edge(node1, node2, attrs={"weight": 0.2})  # node1 -> node2
    edge2 = graph_backend.add_edge(node2, node3, attrs={"weight": 0.3})  # node2 -> node3
    edge3 = graph_backend.add_edge(node3, node0, attrs={"weight": 0.4})  # node3 -> node0

    print(f"Created edges: {edge0=}, {edge1=}, {edge2=}, {edge3=}")

    # Get all edges for reference
    df_all = graph_backend.edge_attrs()
    print(f"All edges:\n{df_all}")

    # Test with include_targets=False (default)
    # When selecting [node1, node2, node3], should only include edges between these nodes:
    # - edge0: node0 -> node1 ✗ (node0 not in selection)
    # - edge1: node1 -> node2 ✓
    # - edge2: node2 -> node3 ✓
    # - edge3: node3 -> node0 ✗ (node0 not in selection)
    df_exclusive = graph_backend.edge_attrs(node_ids=[node1, node2, node3], include_targets=False)
    print(f"Exclusive edges (include_targets=False):\n{df_exclusive}")
    exclusive_edge_ids = set(df_exclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_exclusive = {edge1, edge2}

    print(f"Expected exclusive edge IDs: {expected_exclusive}")
    print(f"Actual exclusive edge IDs: {exclusive_edge_ids}")

    msg = f"include_targets=False: Expected {expected_exclusive}, got {exclusive_edge_ids}"
    assert exclusive_edge_ids == expected_exclusive, msg

    # Verify the weights match expected edges
    exclusive_weights = df_exclusive["weight"].to_list()
    expected_weights = [0.2, 0.3]  # weights for edge1, edge2
    assert sorted(exclusive_weights) == sorted(expected_weights)

    # Test with include_targets=True
    # When selecting [node2, node3], should include edges to neighbors:
    # - edge0: node0 -> node1 ✗ (node0 not in selection)
    # - edge1: node1 -> node2 ✗ (node1 not in selection)
    # - edge2: node2 -> node3 ✓
    # - edge3: node3 -> node0 ✓
    df_inclusive = graph_backend.edge_attrs(node_ids=[node2, node3], include_targets=True)
    print(f"Inclusive edges (include_targets=True):\n{df_inclusive}")
    inclusive_edge_ids = set(df_inclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_inclusive = {edge2, edge3}

    print(f"Expected inclusive edge IDs: {expected_inclusive}")
    print(f"Actual inclusive edge IDs: {inclusive_edge_ids}")

    msg = f"include_targets=True: Expected {expected_inclusive}, got {inclusive_edge_ids}"
    assert inclusive_edge_ids == expected_inclusive, msg

    # Verify all weights are included
    inclusive_weights = df_inclusive["weight"].to_list()
    expected_all_weights = [0.3, 0.4]  # weights for all edges
    assert sorted(inclusive_weights) == sorted(expected_all_weights)

    # Test edge case: selecting only one node with include_targets=True
    # When selecting [node1], with include_targets=True should include edges to neighbors:
    # - edge0: node0 -> node1 ✗ (node1 not in selection)
    # - edge1: node1 -> node2 ✓
    # - edge2: node2 -> node3 ✗ (node1 not in selection)
    # - edge3: node3 -> node0 ✗ (node1 not in selection)
    df_single_inclusive = graph_backend.edge_attrs(node_ids=[node1], include_targets=True)
    print(f"Single node inclusive edges: {df_single_inclusive}")
    single_inclusive_edge_ids = set(df_single_inclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_single_inclusive = {edge1}

    msg = f"Single node include_targets=True: Expected {expected_single_inclusive}, got {single_inclusive_edge_ids}"
    assert single_inclusive_edge_ids == expected_single_inclusive, msg

    # Test edge case: selecting only one node with include_targets=False
    # When selecting [node1], with include_targets=False should include no edges
    # (since there are no edges strictly between just node1)
    df_single_exclusive = graph_backend.edge_attrs(node_ids=[node1], include_targets=False)
    print(f"Single node exclusive edges: {df_single_exclusive}")
    single_exclusive_edge_ids = set(df_single_exclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_single_exclusive = set()  # No edges strictly within [node1]

    msg = f"Single node include_targets=False: Expected {expected_single_exclusive}, got {single_exclusive_edge_ids}"
    assert single_exclusive_edge_ids == expected_single_exclusive, msg


def test_from_ctc(
    ctc_data_dir: Path,
    graph_backend: BaseGraph,
) -> None:
    # ctc data comes from
    # https://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip

    if isinstance(graph_backend, SQLGraph):
        kwargs = {"drivername": "sqlite", "database": ":memory:", "overwrite": True}
    else:
        kwargs = {}

    graph = graph_backend.__class__.from_ctc(ctc_data_dir / "02_GT/TRA", **kwargs)

    assert graph.num_nodes > 0
    assert graph.num_edges > 0


def test_sucessors_and_degree(graph_backend: BaseGraph) -> None:
    """Test getting successors of nodes."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create a simple graph structure: node0 -> node1 -> node2
    #                                      \-> node3
    node0 = graph_backend.add_node({"t": 0, "x": 0.0, "y": 0.0})
    node1 = graph_backend.add_node({"t": 1, "x": 1.0, "y": 1.0})
    node2 = graph_backend.add_node({"t": 2, "x": 2.0, "y": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0, "y": 3.0})

    # Add edges
    graph_backend.add_edge(node0, node1, {"weight": 0.5})  # node0 -> node1
    graph_backend.add_edge(node0, node3, {"weight": 0.7})  # node0 -> node3
    graph_backend.add_edge(node1, node2, {"weight": 0.3})  # node1 -> node2

    # Test successors of node0 (should return node1 and node3)
    successors_df = graph_backend.successors(node0)
    assert isinstance(successors_df, pl.DataFrame)
    assert len(successors_df) == 2  # node0 has 2 successors
    assert graph_backend.out_degree(node0) == 2

    # Check that we get the correct target nodes (order doesn't matter)
    successor_nodes = set(successors_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list())
    assert successor_nodes == {node1, node3}

    # Test successors of node1 (should return node2)
    successors_df = graph_backend.successors(node1)
    assert isinstance(successors_df, pl.DataFrame)
    assert len(successors_df) == 1  # node1 has 1 successor
    assert successors_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()[0] == node2
    assert graph_backend.out_degree(node1) == 1

    # Test successors of node2 (should return empty - no successors)
    successors_df = graph_backend.successors(node2)
    assert isinstance(successors_df, pl.DataFrame)
    assert len(successors_df) == 0  # node2 has no successors
    assert graph_backend.out_degree(node2) == 0

    # Test with multiple nodes
    successors_dict = graph_backend.successors([node0, node1, node2])
    assert isinstance(successors_dict, dict)
    assert len(successors_dict) == 3

    # testing query all
    assert graph_backend.out_degree() == [2, 1, 0, 0]

    # testing different ordering
    assert graph_backend.out_degree([node0, node1, node2]) == [2, 1, 0]
    assert graph_backend.out_degree([node1, node2, node0]) == [1, 0, 2]

    # Check node0's successors
    assert len(successors_dict[node0]) == 2
    # Check node1's successors
    assert len(successors_dict[node1]) == 1
    # Check node2's successors (empty)
    assert len(successors_dict[node2]) == 0


def test_predecessors_and_degree(graph_backend: BaseGraph) -> None:
    """Test getting predecessors of nodes."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create a simple graph structure: node0 -> node1 -> node2
    #                                      \-> node3
    node0 = graph_backend.add_node({"t": 0, "x": 0.0, "y": 0.0})
    node1 = graph_backend.add_node({"t": 1, "x": 1.0, "y": 1.0})
    node2 = graph_backend.add_node({"t": 2, "x": 2.0, "y": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0, "y": 3.0})

    # Add edges
    graph_backend.add_edge(node0, node1, {"weight": 0.5})  # node0 -> node1
    graph_backend.add_edge(node0, node3, {"weight": 0.7})  # node0 -> node3
    graph_backend.add_edge(node1, node2, {"weight": 0.3})  # node1 -> node2

    # Test predecessors of node0 (should return empty - no predecessors)
    predecessors_df = graph_backend.predecessors(node0)
    assert isinstance(predecessors_df, pl.DataFrame)
    assert len(predecessors_df) == 0  # node0 has no predecessors
    assert graph_backend.in_degree(node0) == 0

    # Test predecessors of node1 (should return node0)
    predecessors_df = graph_backend.predecessors(node1)
    assert isinstance(predecessors_df, pl.DataFrame)
    assert len(predecessors_df) == 1  # node1 has 1 predecessor
    assert graph_backend.in_degree(node1) == 1

    # Check that we get the correct source node
    assert predecessors_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()[0] == node0

    # Test predecessors of node2 (should return node1)
    predecessors_df = graph_backend.predecessors(node2)
    assert isinstance(predecessors_df, pl.DataFrame)
    assert len(predecessors_df) == 1  # node2 has 1 predecessor
    assert predecessors_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()[0] == node1
    assert graph_backend.in_degree(node2) == 1

    # Test predecessors of node3 (should return node0)
    predecessors_df = graph_backend.predecessors(node3)
    assert isinstance(predecessors_df, pl.DataFrame)
    assert len(predecessors_df) == 1  # node3 has 1 predecessor
    assert predecessors_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()[0] == node0
    assert graph_backend.in_degree(node3) == 1

    # Test with multiple nodes
    predecessors_dict = graph_backend.predecessors([node0, node1, node2, node3])
    assert isinstance(predecessors_dict, dict)
    assert len(predecessors_dict) == 4
    assert graph_backend.in_degree() == [0, 1, 1, 1]
    # testing different ordering
    assert graph_backend.in_degree([node0, node1, node2, node3]) == [0, 1, 1, 1]
    assert graph_backend.in_degree([node1, node2, node3, node0]) == [1, 1, 1, 0]

    # Check predecessors
    assert len(predecessors_dict[node0]) == 0  # node0 has no predecessors
    assert len(predecessors_dict[node1]) == 1  # node1 has 1 predecessor
    assert len(predecessors_dict[node2]) == 1  # node2 has 1 predecessor
    assert len(predecessors_dict[node3]) == 1  # node3 has 1 predecessor


def test_sucessors_with_attr_keys(graph_backend: BaseGraph) -> None:
    """Test getting successors with specific attribute keys."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_node_attr_key("label", "X")
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create nodes
    node0 = graph_backend.add_node({"t": 0, "x": 0.0, "y": 0.0, "label": "A"})
    node1 = graph_backend.add_node({"t": 1, "x": 1.0, "y": 1.0, "label": "B"})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0, "label": "C"})

    # Add edges
    graph_backend.add_edge(node0, node1, {"weight": 0.5})
    graph_backend.add_edge(node0, node2, {"weight": 0.7})

    # Test with single attribute key as string
    successors_df = graph_backend.successors(node0, attr_keys="x")
    assert isinstance(successors_df, pl.DataFrame)
    assert "x" in successors_df.columns
    assert "y" not in successors_df.columns

    # Should not contain other attribute keys when we specify specific ones
    available_cols = set(successors_df.columns)
    # The exact columns depend on implementation, but x should be there
    assert "x" in available_cols

    # Test with multiple attribute keys as list
    successors_df = graph_backend.successors(node0, attr_keys=["x", "label"])
    assert isinstance(successors_df, pl.DataFrame)
    assert "x" in successors_df.columns
    assert "label" in successors_df.columns
    assert "y" not in successors_df.columns

    # Verify the content makes sense
    if len(successors_df) > 0:
        x_values = successors_df["x"].to_list()
        label_values = successors_df["label"].to_list()
        # These should correspond to node1 and node2's attributes
        assert set(x_values) == {1.0, 2.0}
        assert set(label_values) == {"B", "C"}


def test_predecessors_with_attr_keys(graph_backend: BaseGraph) -> None:
    """Test getting predecessors with specific attribute keys."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_node_attr_key("label", "X")
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create nodes
    node0 = graph_backend.add_node({"t": 0, "x": 0.0, "y": 0.0, "label": "A"})
    node1 = graph_backend.add_node({"t": 0, "x": 1.0, "y": 1.0, "label": "B"})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0, "label": "C"})

    # Add edges (both node0 and node1 point to node2)
    graph_backend.add_edge(node0, node2, {"weight": 0.5})
    graph_backend.add_edge(node1, node2, {"weight": 0.7})

    # Test with single attribute key as string
    predecessors_df = graph_backend.predecessors(node2, attr_keys="label")
    assert isinstance(predecessors_df, pl.DataFrame)
    assert "label" in predecessors_df.columns
    assert "y" not in predecessors_df.columns
    assert "x" not in predecessors_df.columns

    # Test with multiple attribute keys as list
    predecessors_df = graph_backend.predecessors(node2, attr_keys=["x", "label"])
    assert isinstance(predecessors_df, pl.DataFrame)
    assert "x" in predecessors_df.columns
    assert "label" in predecessors_df.columns
    assert "y" not in predecessors_df.columns

    # Verify the content makes sense - should have 2 predecessors
    assert len(predecessors_df) == 2
    x_values = predecessors_df["x"].to_list()
    label_values = predecessors_df["label"].to_list()
    # These should correspond to node0 and node1's attributes
    assert set(x_values) == {0.0, 1.0}
    assert set(label_values) == {"A", "B"}


def test_sucessors_predecessors_edge_cases(graph_backend: BaseGraph) -> None:
    """Test edge cases for successors and predecessors methods."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Create isolated nodes (no edges)
    node0 = graph_backend.add_node({"t": 0, "x": 0.0})
    node1 = graph_backend.add_node({"t": 1, "x": 1.0})

    # Test successors/predecessors of isolated nodes
    successors_df = graph_backend.successors(node0)
    assert isinstance(successors_df, pl.DataFrame)
    assert len(successors_df) == 0

    predecessors_df = graph_backend.predecessors(node1)
    assert isinstance(predecessors_df, pl.DataFrame)
    assert len(predecessors_df) == 0

    # Test with empty list of nodes
    successors_dict = graph_backend.successors([])
    assert isinstance(successors_dict, dict)
    assert len(successors_dict) == 0

    predecessors_dict = graph_backend.predecessors([])
    assert isinstance(predecessors_dict, dict)
    assert len(predecessors_dict) == 0

    # Test with non-existent attribute keys (should work but return limited columns)
    # This depends on implementation - some might raise errors, others might ignore
    try:
        successors_df = graph_backend.successors(node0, attr_keys=["nonexistent"])
        # If it doesn't raise an error, it should return empty or handle gracefully
        assert isinstance(successors_df, pl.DataFrame)
    except (KeyError, AttributeError):
        # This is also acceptable behavior
        pass


def test_match_method(graph_backend: BaseGraph) -> None:
    """Test the match method for matching nodes between two graphs."""
    # Create first graph (self) with masks
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create masks for first graph
    mask1_data = np.array([[True, True], [True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, False], [True, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([10, 10, 12, 12]))

    mask3_data = np.array([[True, True, True, True, True]], dtype=bool)
    mask3 = Mask(mask3_data, bbox=np.array([20, 20, 21, 25]))

    # Add nodes to first graph
    node1 = graph_backend.add_node({"t": 0, "x": 1.0, "y": 1.0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0, DEFAULT_ATTR_KEYS.MASK: mask2})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0, "y": 3.0, DEFAULT_ATTR_KEYS.MASK: mask3})

    # Add edges to first graph
    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, {"weight": 0.5})
    graph_backend.add_edge(node2, node3, {"weight": 0.3})

    # this edge won't be matched
    graph_backend.add_edge(node1, node3, {"weight": 0.3})

    # Create second graph (other/reference) with overlapping masks
    if isinstance(graph_backend, SQLGraph):
        kwargs = {"drivername": "sqlite", "database": ":memory:"}
    else:
        kwargs = {}

    other_graph = graph_backend.__class__(**kwargs)
    other_graph.add_node_attr_key("x", 0.0)
    other_graph.add_node_attr_key("y", 0.0)
    other_graph.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create overlapping masks for second graph
    # This mask overlaps significantly with mask1 (IoU > 0.5)
    ref_mask1_data = np.array([[True, True], [True, False]], dtype=bool)
    ref_mask1 = Mask(ref_mask1_data, bbox=np.array([0, 0, 2, 2]))

    # This mask should NOT overlap with mask2 (IoU < 0.5, should not match)
    ref_mask2_data = np.array([[True]], dtype=bool)
    ref_mask2 = Mask(ref_mask2_data, bbox=np.array([15, 15, 16, 16]))  # Different location

    # This mask overlaps significantly with mask3 (IoU > 0.5)
    ref_mask3_data = np.array([[True, True, True, True]], dtype=bool)
    ref_mask3 = Mask(ref_mask3_data, bbox=np.array([20, 20, 21, 24]))

    # This mask also overlaps significantly with mask3 (IoU > 0.5) but less than `ref_mask3`
    # therefore it should not match
    ref_mask4_data = np.array([[True, True, True]], dtype=bool)
    ref_mask4 = Mask(ref_mask4_data, bbox=np.array([20, 21, 21, 24]))

    # Add nodes to reference graph
    ref_node1 = other_graph.add_node({"t": 0, "x": 1.1, "y": 1.1, DEFAULT_ATTR_KEYS.MASK: ref_mask1})
    ref_node2 = other_graph.add_node({"t": 1, "x": 2.1, "y": 2.1, DEFAULT_ATTR_KEYS.MASK: ref_mask2})
    ref_node3 = other_graph.add_node({"t": 2, "x": 3.1, "y": 3.1, DEFAULT_ATTR_KEYS.MASK: ref_mask3})
    ref_node4 = other_graph.add_node({"t": 2, "x": 3.1, "y": 3.1, DEFAULT_ATTR_KEYS.MASK: ref_mask4})

    # Add edges to reference graph - matching structure with first graph
    other_graph.add_edge_attr_key("weight", 0.0)
    other_graph.add_edge(ref_node1, ref_node2, {"weight": 0.6})  # ref_node1 -> ref_node2
    other_graph.add_edge(ref_node2, ref_node3, {"weight": 0.7})  # ref_node2 -> ref_node3
    other_graph.add_edge(ref_node2, ref_node4, {"weight": 0.5})  # ref_node3 -> ref_node4

    # Test the match method
    match_node_id_key = "matched_node_id"
    match_score_key = "match_score"
    edge_match_key = "edge_matched"

    graph_backend.match(
        other=other_graph,
        matched_node_id_key=match_node_id_key,
        match_score_key=match_score_key,
        matched_edge_mask_key=edge_match_key,
    )

    # Verify that attribute keys were added
    assert match_node_id_key in graph_backend.node_attr_keys
    assert match_score_key in graph_backend.node_attr_keys
    assert edge_match_key in graph_backend.edge_attr_keys

    # Get node attributesto check matching results
    nodes_df = graph_backend.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.NODE_ID, match_node_id_key, match_score_key])
    print(nodes_df)

    # Verify specific expected matches based on IoU
    # Create a mapping from node_id to matched values
    node_matches = {}
    for row in nodes_df.iter_rows(named=True):
        node_matches[row[DEFAULT_ATTR_KEYS.NODE_ID]] = {
            "matched_id": row[match_node_id_key],
            "score": row[match_score_key],
        }

    assert len(nodes_df) == graph_backend.num_nodes

    # Check expected matches:
    # node1 (mask1) should match ref_node1 (ref_mask1) - high IoU
    msg = f"node1 should match ref_node1, got {node_matches[node1]['matched_id']}"
    assert node_matches[node1]["matched_id"] == ref_node1, msg
    msg = f"node1 match score should be > 0.5, got {node_matches[node1]['score']}"
    assert node_matches[node1]["score"] > 0.5, msg

    # node2 (mask2) should NOT match ref_node2 (ref_mask2) - low IoU
    msg = f"node2 should not match (should be -1), got {node_matches[node2]['matched_id']}"
    assert node_matches[node2]["matched_id"] == -1, msg
    msg = f"node2 match score should be 0.0, got {node_matches[node2]['score']}"
    assert node_matches[node2]["score"] == 0.0, msg

    # node3 (mask3) should match ref_node3 (ref_mask3) - high IoU
    msg = f"node3 should match ref_node3, got {node_matches[node3]['matched_id']}"
    assert node_matches[node3]["matched_id"] == ref_node3, msg
    msg = f"node3 match score should be > 0.5, got {node_matches[node3]['score']}"
    assert node_matches[node3]["score"] > 0.5, msg

    # Verify match scores are reasonable (between 0 and 1)
    for node_id, match_info in node_matches.items():
        score = match_info["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} for node {node_id} should be between 0 and 1"

    # Check edge matching
    edges_df = graph_backend.edge_attrs(attr_keys=[edge_match_key])
    assert len(edges_df) > 0

    # After your bug fixes, both edges are matching
    edge_matches = edges_df[edge_match_key].to_list()
    expected_matches = np.array([True, True, False])

    np.testing.assert_array_equal(edge_matches, expected_matches)


def test_attrs_with_duplicated_attr_keys(graph_backend: BaseGraph) -> None:
    """Test that node attributeswith duplicated attribute keys are handled correctly."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)

    # Add nodes
    graph_backend.add_node({"t": 0, "x": 1.0, "y": 1.0})
    graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0})

    # Add edges
    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge(0, 1, {"weight": 0.5})

    # Test with duplicated attribute keys
    # This would crash before
    nodes_df = graph_backend.node_attrs(attr_keys=["x", "y", "x"])
    assert "x" in nodes_df.columns
    assert "y" in nodes_df.columns
    assert nodes_df["x"].to_list() == [1.0, 2.0]
    assert nodes_df["y"].to_list() == [1.0, 2.0]

    edges_df = graph_backend.edge_attrs(attr_keys=["weight", "weight", "weight"])
    assert "weight" in edges_df.columns
    assert edges_df["weight"].to_list() == [0.5]


def test_add_overlap(graph_backend: BaseGraph) -> None:
    """Test adding single overlaps to the graph."""
    # Add nodes first
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 0})
    node3 = graph_backend.add_node({"t": 1})

    # Add overlaps
    graph_backend.add_overlap(node1, node2)
    graph_backend.add_overlap(node2, node3)

    # Verify overlaps were added
    assert graph_backend.has_overlaps()
    overlaps = graph_backend.overlaps()
    assert len(overlaps) == 2
    assert [node1, node2] in overlaps
    assert [node2, node3] in overlaps


def test_bulk_add_overlaps(graph_backend: BaseGraph) -> None:
    """Test adding multiple overlaps efficiently."""
    # Add nodes first
    nodes = []
    for i in range(5):
        nodes.append(graph_backend.add_node({"t": i}))

    # Create overlap pairs
    overlap_pairs = [
        [nodes[0], nodes[1]],
        [nodes[1], nodes[2]],
        [nodes[2], nodes[3]],
        [nodes[3], nodes[4]],
    ]

    # Add overlaps in bulk
    graph_backend.bulk_add_overlaps(overlap_pairs)

    # Verify all overlaps were added
    assert graph_backend.has_overlaps()
    overlaps = graph_backend.overlaps()
    assert len(overlaps) == 4
    for pair in overlap_pairs:
        assert pair in overlaps


def test_overlaps_with_node_filtering(graph_backend: BaseGraph) -> None:
    """Test retrieving overlaps filtered by specific node IDs."""
    # Add nodes
    nodes = []
    for i in range(4):
        nodes.append(graph_backend.add_node({"t": i}))

    # Add overlaps
    graph_backend.add_overlap(nodes[0], nodes[1])
    graph_backend.add_overlap(nodes[1], nodes[2])
    graph_backend.add_overlap(nodes[2], nodes[3])

    # Test filtering by nodes that have multiple overlaps
    filtered_overlaps = graph_backend.overlaps([nodes[1], nodes[2]])
    assert len(filtered_overlaps) == 1
    assert [nodes[1], nodes[2]] == filtered_overlaps[0]

    # Test filtering by nodes with no overlaps
    filtered_overlaps = graph_backend.overlaps([nodes[0], nodes[3]])
    assert len(filtered_overlaps) == 0


def test_overlaps_empty_graph(graph_backend: BaseGraph) -> None:
    """Test overlap behavior on empty graphs."""
    # Test on empty graph
    assert not graph_backend.has_overlaps()
    assert graph_backend.overlaps() == []
    assert graph_backend.overlaps([1, 2, 3]) == []


def test_overlaps_edge_cases(graph_backend: BaseGraph) -> None:
    """Test overlap functionality with edge cases."""
    # Add a single node
    node = graph_backend.add_node({"t": 0})

    # Test overlaps with single node (should be empty)
    assert graph_backend.overlaps([node]) == []

    # Test overlaps with non-existent nodes
    assert graph_backend.overlaps([999, 1000]) == []

    # Add overlap and test with mixed existing/non-existing nodes
    node2 = graph_backend.add_node({"t": 0})
    graph_backend.add_overlap(node, node2)

    overlaps = graph_backend.overlaps([node, 999])
    assert len(overlaps) == 0


def test_from_numpy_array_basic(graph_backend: BaseGraph) -> None:
    """Test basic functionality of from_numpy_array with 2D positions."""
    # Test 2D positions (T, Y, X)
    positions = np.array(
        [
            [0, 10, 20],  # t=0, y=10, x=20
            [1, 15, 25],  # t=1, y=15, x=25
            [2, 20, 30],  # t=2, y=20, x=30
        ]
    )

    radius = 2

    if isinstance(graph_backend, RustWorkXGraph):
        # for RustWorkXGraph we validate if the OOP API is working
        graph_backend = RustWorkXGraph.from_array(positions, radius=radius, rx_graph=None)
    else:
        load_array(positions, graph_backend, radius=radius)

    assert graph_backend.num_nodes == 3
    assert graph_backend.num_edges == 0  # No track_ids, so no edges

    # Check node attributes
    nodes_df = graph_backend.node_attrs(attr_keys=["t", "y", "x"])

    np.testing.assert_array_equal(nodes_df.to_numpy(), positions)

    # Check that mask attribute was added
    assert DEFAULT_ATTR_KEYS.MASK in graph_backend.node_attr_keys


def test_from_numpy_array_3d(graph_backend: BaseGraph) -> None:
    """Test from_numpy_array with 3D positions (T, Z, Y, X)."""
    # Test 3D positions (T, Z, Y, X)
    positions = np.asarray(
        [
            [0, 5, 10, 20],  # t=0, z=5, y=10, x=20
            [1, 6, 15, 25],  # t=1, z=6, y=15, x=25
            [2, 7, 20, 30],  # t=2, z=7, y=20, x=30
        ]
    )

    track_ids = np.asarray([1, 2, 3])
    track_id_graph = {3: 1, 2: 1}

    radius = np.asarray([1, 3, 5])

    if isinstance(graph_backend, RustWorkXGraph):
        # for RustWorkXGraph we validate if the OOP API is working
        graph_backend = RustWorkXGraph.from_array(
            positions,
            track_ids=track_ids,
            track_id_graph=track_id_graph,
            radius=radius,
            rx_graph=None,
        )
    else:
        load_array(
            positions,
            graph_backend,
            track_ids=track_ids,
            track_id_graph=track_id_graph,
            radius=radius,
        )

    assert graph_backend.num_nodes == 3
    assert graph_backend.num_edges == 2

    edges_df = graph_backend.edge_attrs()
    assert len(edges_df) == 2

    nodes_df = graph_backend.node_attrs()
    node_ids = nodes_df[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

    edges = edges_df.select([DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]).to_numpy().tolist()
    assert [node_ids[0], node_ids[1]] in edges
    assert [node_ids[0], node_ids[2]] in edges

    np.testing.assert_array_equal(nodes_df.select(["t", "z", "y", "x"]).to_numpy(), positions)

    masks = [m.bbox[3] - m.bbox[0] for m in nodes_df[DEFAULT_ATTR_KEYS.MASK].to_list()]
    np.testing.assert_array_equal(masks, [r * 2 + 1 for r in radius])

    np.testing.assert_array_equal(nodes_df[DEFAULT_ATTR_KEYS.TRACK_ID].to_list(), track_ids)


def test_from_numpy_array_validation_errors() -> None:
    """Test from_numpy_array validation errors."""
    # Test invalid position dimensions
    invalid_positions = np.array([[0, 10]])  # Only 2 columns, need 3 or 4
    with pytest.raises(ValueError, match="Expected 4 or 5 dimensions"):
        RustWorkXGraph.from_array(invalid_positions)

    # Test radius length mismatch
    positions = np.array([[0, 10, 20], [1, 15, 25]])
    invalid_radius = np.array([1, 2, 3])  # Length 3, positions length 2
    with pytest.raises(ValueError, match="must be a scalar or have the same length"):
        RustWorkXGraph.from_array(positions, radius=invalid_radius)

    # Test track_id_graph without track_ids
    with pytest.raises(ValueError, match="must be provided if"):
        RustWorkXGraph.from_array(positions, track_id_graph={2: 1})

    # Test track_ids length mismatch
    track_ids = np.array([1, 2, 3])  # Length 3, positions length 2
    with pytest.raises(ValueError, match="must have the same length"):
        RustWorkXGraph.from_array(positions, track_ids=track_ids)
