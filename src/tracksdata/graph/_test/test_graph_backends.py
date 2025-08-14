from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rustworkx as rx
from zarr.storage import MemoryStore

from tracksdata.attrs import EdgeAttr, NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import BaseGraph, RustWorkXGraph, SQLGraph
from tracksdata.io._numpy_array import from_array
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
    with pytest.raises((ValueError, KeyError)):
        graph_backend.add_edge(0, 1, {"weight": 0.5})


def test_add_node(graph_backend: BaseGraph) -> None:
    """Test adding nodes with various attributes."""

    for key in ["x", "y"]:
        graph_backend.add_node_attr_key(key, 0.0)

    node_id = graph_backend.add_node({"t": 0, "x": 1.0, "y": 2.0})
    assert isinstance(node_id, int)

    # Check node attributes
    df = graph_backend.filter(node_ids=[node_id]).node_attrs()
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

    df = graph_backend.edge_attrs()
    assert df["new_attribute"].to_list() == [0.0, 1.0]
    assert df["weight"].to_list() == [0.5, 0.1]


def test_node_ids(graph_backend: BaseGraph) -> None:
    """Test retrieving node IDs."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 1})

    assert len(graph_backend.node_ids()) == 2


def test_filter_nodes_by_attribute(graph_backend: BaseGraph) -> None:
    """Test filtering nodes by attributes."""
    graph_backend.add_node_attr_key("label", None)

    node1 = graph_backend.add_node({"t": 0, "label": "A"})
    node2 = graph_backend.add_node({"t": 0, "label": "B"})
    node3 = graph_backend.add_node({"t": 1, "label": "A"})

    # Filter by time
    nodes = graph_backend.filter(NodeAttr("t") == 0).node_ids()
    assert set(nodes) == {node1, node2}

    # Filter by label
    nodes = graph_backend.filter(NodeAttr("label") == "A").node_ids()
    assert set(nodes) == {node1, node3}

    # Filter by t and label using multiple conditions
    nodes = graph_backend.filter(NodeAttr("t") == 1, NodeAttr("label") == "A").node_ids()
    assert set(nodes) == {node3}

    # Test with inequality
    nodes = graph_backend.filter(NodeAttr("t") > 0).node_ids()
    assert set(nodes) == {node3}

    # Test with multiple conditions using *args for AND
    nodes = graph_backend.filter(NodeAttr("t") == 0, NodeAttr("label") == "A").node_ids()
    assert set(nodes) == {node1}


def test_time_points(graph_backend: BaseGraph) -> None:
    """Test retrieving time points."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 2})
    graph_backend.add_node({"t": 1})

    assert set(graph_backend.time_points()) == {0, 1, 2}


def test_node_attrs(graph_backend: BaseGraph) -> None:
    """Test retrieving node attributes."""
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("coordinates", np.array([0.0, 0.0]))

    node1 = graph_backend.add_node({"t": 0, "x": 1.0, "coordinates": np.array([10.0, 20.0])})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "coordinates": np.array([30.0, 40.0])})

    df = graph_backend.filter(node_ids=[node1, node2]).node_attrs(attr_keys=["x"])
    assert isinstance(df, pl.DataFrame)
    assert df["x"].to_list() == [1.0, 2.0]

    # Test unpack functionality
    df_unpacked = graph_backend.filter(node_ids=[node1, node2]).node_attrs(attr_keys=["coordinates"], unpack=True)
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
    df_subset = graph_backend.filter(node_ids=[node1, node2, node3]).edge_attrs()
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
    assert expected_subset_edge_ids == actual_subset_edge_ids, msg


def test_subgraph_with_node_and_edge_attr_filters(graph_backend: BaseGraph) -> None:
    """Test subgraph with node and edge attribute filters."""
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 1.0})
    node4 = graph_backend.add_node({"t": 3, "x": 3.0})
    node5 = graph_backend.add_node({"t": 4, "x": 0.5})

    graph_backend.add_edge(node1, node3, attrs={"weight": 0.8})
    edge2 = graph_backend.add_edge(node3, node5, attrs={"weight": 0.2})
    graph_backend.add_edge(node2, node4, attrs={"weight": 0.0})

    subgraph = graph_backend.filter(
        NodeAttr("x") <= 1.0,
        EdgeAttr("weight") < 0.5,
    ).subgraph()

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

    node0 = graph_backend.add_node({"t": 0, "x": 1.0})
    node1 = graph_backend.add_node({"t": 1, "x": 2.0})
    node2 = graph_backend.add_node({"t": 2, "x": 1.0})
    node3 = graph_backend.add_node({"t": 3, "x": 3.0})
    node4 = graph_backend.add_node({"t": 4, "x": 0.5})

    graph_backend.add_edge(node0, node2, attrs={"weight": 0.8})
    graph_backend.add_edge(node2, node4, attrs={"weight": 0.2})
    graph_backend.add_edge(node1, node3, attrs={"weight": 0.0})

    g_filter = graph_backend.filter(
        NodeAttr("x") <= 1.0,
        EdgeAttr("weight") < 0.5,
        node_ids=[node0, node2],
    )
    node_ids = g_filter.node_ids()
    assert set(node_ids) == {node0, node2}

    subgraph = g_filter.subgraph()

    assert subgraph.num_nodes == 2
    assert subgraph.num_edges == 0

    subgraph_node_ids = subgraph.node_ids()
    assert set(subgraph_node_ids) == {node0, node2}

    subgraph_edge_ids = subgraph.edge_ids()
    assert len(subgraph_edge_ids) == 0


def test_add_node_attr_key(graph_backend: BaseGraph) -> None:
    """Test adding new node attribute keys."""
    node = graph_backend.add_node({"t": 0})
    graph_backend.add_node_attr_key("new_attribute", 42)

    df = graph_backend.filter(node_ids=[node]).node_attrs(attr_keys=["new_attribute"])
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
    graph_backend.add_node_attr_key("x", 0.0)

    node_1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node_2 = graph_backend.add_node({"t": 0, "x": 2.0})

    graph_backend.update_node_attrs(node_ids=[node_1], attrs={"x": 3.0})

    df = graph_backend.filter(node_ids=[node_1, node_2]).node_attrs(attr_keys=["x"])
    assert df["x"].to_list() == [3.0, 2.0]

    # inverted access on purpose
    graph_backend.update_node_attrs(node_ids=[node_2, node_1], attrs={"x": [5.0, 6.0]})

    df = graph_backend.filter(node_ids=[node_1, node_2]).node_attrs(attr_keys=["x"])
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
    df = graph_backend.filter(node_ids=[node1, node2]).edge_attrs(attr_keys=["weight"])
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
    df_exclusive = graph_backend.filter(node_ids=[node1, node2, node3], include_targets=False).edge_attrs()
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
    df_inclusive = graph_backend.filter(node_ids=[node2, node3], include_targets=True).edge_attrs()
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
    df_single_inclusive = graph_backend.filter(node_ids=[node1], include_targets=True).edge_attrs()
    print(f"Single node inclusive edges: {df_single_inclusive}")
    single_inclusive_edge_ids = set(df_single_inclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_single_inclusive = {edge1}

    msg = f"Single node include_targets=True: Expected {expected_single_inclusive}, got {single_inclusive_edge_ids}"
    assert single_inclusive_edge_ids == expected_single_inclusive, msg

    # Test edge case: selecting only one node with include_targets=False
    # When selecting [node1], with include_targets=False should include no edges
    # (since there are no edges strictly between just node1)
    df_single_exclusive = graph_backend.filter(node_ids=[node1], include_targets=False).edge_attrs()
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

    graph_backend.add_edge_attr_key("weight", 0.0)
    # this will not be matched
    graph_backend.add_edge(node1, node2, {"weight": 0.5})
    graph_backend.add_edge(node2, node3, {"weight": 0.3})

    # this will be matched
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

    # This mask overlaps significantly with mask3 (IoU > 0.5)
    ref_mask2_data = np.array([[True, True, True, True]], dtype=bool)
    ref_mask2 = Mask(ref_mask2_data, bbox=np.array([20, 20, 21, 24]))

    # This mask should NOT overlap with other masks (IoU < 0.5, should not match)
    ref_mask3_data = np.array([[True]], dtype=bool)
    ref_mask3 = Mask(ref_mask3_data, bbox=np.array([15, 15, 16, 16]))  # Different location

    # This mask also overlaps significantly with mask3 (IoU > 0.5) but less than `ref_mask2`
    # therefore it should not match
    ref_mask4_data = np.array([[True, True, True]], dtype=bool)
    ref_mask4 = Mask(ref_mask4_data, bbox=np.array([20, 21, 21, 24]))

    # Add nodes to reference graph
    ref_node1 = other_graph.add_node({"t": 0, "x": 1.1, "y": 1.1, DEFAULT_ATTR_KEYS.MASK: ref_mask1})
    ref_node2 = other_graph.add_node({"t": 2, "x": 3.1, "y": 3.1, DEFAULT_ATTR_KEYS.MASK: ref_mask2})
    ref_node3 = other_graph.add_node({"t": 1, "x": 2.1, "y": 2.1, DEFAULT_ATTR_KEYS.MASK: ref_mask3})
    ref_node4 = other_graph.add_node({"t": 2, "x": 3.1, "y": 3.1, DEFAULT_ATTR_KEYS.MASK: ref_mask4})

    # Add edges to reference graph - matching structure with first graph
    other_graph.add_edge_attr_key("weight", 0.0)
    other_graph.add_edge(ref_node1, ref_node3, {"weight": 0.6})  # ref_node1 -> ref_node2
    other_graph.add_edge(ref_node1, ref_node2, {"weight": 0.4})  # ref_node1 -> ref_node3
    other_graph.add_edge(ref_node3, ref_node2, {"weight": 0.7})  # ref_node2 -> ref_node3
    other_graph.add_edge(ref_node3, ref_node4, {"weight": 0.5})  # ref_node3 -> ref_node4

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

    # node3 (mask3) should match ref_node2 (ref_mask2) - high IoU
    msg = f"node3 should match ref_node2, got {node_matches[node3]['matched_id']}"
    assert node_matches[node3]["matched_id"] == ref_node2, msg
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
    edge_matches = edges_df.sort(DEFAULT_ATTR_KEYS.EDGE_ID)[edge_match_key].to_list()
    expected_matches = np.asarray([False, False, True])

    np.testing.assert_array_equal(edge_matches, expected_matches)


def test_attrs_with_duplicated_attr_keys(graph_backend: BaseGraph) -> None:
    """Test that node attributeswith duplicated attribute keys are handled correctly."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)

    # Add nodes
    node_1 = graph_backend.add_node({"t": 0, "x": 1.0, "y": 1.0})
    node_2 = graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0})

    # Add edges
    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge(node_1, node_2, {"weight": 0.5})

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

    if isinstance(graph_backend, RustWorkXGraph):
        # for RustWorkXGraph we validate if the OOP API is working
        graph_backend = RustWorkXGraph.from_array(positions, rx_graph=None)
    else:
        from_array(positions, graph_backend)

    assert graph_backend.num_nodes == 3
    assert graph_backend.num_edges == 0  # No track_ids, so no edges

    # Check node attributes
    nodes_df = graph_backend.node_attrs(attr_keys=["t", "y", "x"])

    np.testing.assert_array_equal(nodes_df.to_numpy(), positions)


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

    if isinstance(graph_backend, RustWorkXGraph):
        # for RustWorkXGraph we validate if the OOP API is working
        graph_backend = RustWorkXGraph.from_array(
            positions,
            track_ids=track_ids,
            track_id_graph=track_id_graph,
            rx_graph=None,
        )
    else:
        from_array(
            positions,
            graph_backend,
            track_ids=track_ids,
            track_id_graph=track_id_graph,
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
    np.testing.assert_array_equal(nodes_df[DEFAULT_ATTR_KEYS.TRACK_ID].to_list(), track_ids)


def test_from_numpy_array_validation_errors() -> None:
    """Test from_numpy_array validation errors."""
    # Test invalid position dimensions
    invalid_positions = np.array([[0, 10]])  # Only 2 columns, need 3 or 4
    with pytest.raises(ValueError, match="Expected 4 or 5 dimensions"):
        RustWorkXGraph.from_array(invalid_positions)

    positions = np.array([[0, 10, 20], [1, 15, 25]])

    # Test track_id_graph without track_ids
    with pytest.raises(ValueError, match="must be provided if"):
        RustWorkXGraph.from_array(positions, track_id_graph={2: 1})

    # Test track_ids length mismatch
    track_ids = np.array([1, 2, 3])  # Length 3, positions length 2
    with pytest.raises(ValueError, match="must have the same length"):
        RustWorkXGraph.from_array(positions, track_ids=track_ids)


def test_from_other_with_edges(graph_backend: BaseGraph) -> None:
    """Test from_other method with edges and edge attributes."""
    # Create source graph with nodes, edges, and attributes
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge_attr_key("type", "forward")

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0})

    graph_backend.add_edge(node1, node2, {"weight": 0.5, "type": "forward"})
    graph_backend.add_edge(node2, node3, {"weight": 0.8, "type": "forward"})
    graph_backend.add_edge(node1, node3, {"weight": 0.3, "type": "skip"})

    graph_backend.add_overlap(node1, node3)

    new_graph = RustWorkXGraph.from_other(graph_backend)

    # Verify the new graph has the same structure
    assert new_graph.num_nodes == 3
    assert new_graph.num_edges == 3

    # Verify edge attributes are copied correctly
    source_edges = graph_backend.edge_attrs(attr_keys=["weight", "type"])
    new_edges = new_graph.edge_attrs(attr_keys=["weight", "type"])

    # Edge IDs and node IDs will be different, but edge attributes should be the same
    assert len(source_edges) == len(new_edges)

    # Sort by weight to ensure consistent comparison
    source_sorted = source_edges.sort("weight")
    new_sorted = new_edges.sort("weight")

    assert source_sorted.select(["weight", "type"]).equals(new_sorted.select(["weight", "type"]))

    # Verify attribute keys are preserved
    assert set(new_graph.edge_attr_keys) == set(graph_backend.edge_attr_keys)

    # Verify graph connectivity is preserved by checking degrees
    source_out_degrees = sorted(graph_backend.out_degree())
    new_out_degrees = sorted(new_graph.out_degree())
    assert source_out_degrees == new_out_degrees

    source_in_degrees = sorted(graph_backend.in_degree())
    new_in_degrees = sorted(new_graph.in_degree())
    assert source_in_degrees == new_in_degrees

    new_node_ids = new_graph.node_ids()

    assert len(new_graph.overlaps()) == len(graph_backend.overlaps())
    assert new_graph.overlaps()[0] == [new_node_ids[0], new_node_ids[2]]


def test_compute_overlaps_basic(graph_backend: BaseGraph) -> None:
    """Test basic compute_overlaps functionality."""
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create overlapping masks at time 0
    mask1_data = np.array([[True, True], [True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    mask2_data = np.array([[True, True], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    node1 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask2})

    graph_backend.compute_overlaps(iou_threshold=0.3)

    assert graph_backend.has_overlaps()
    overlaps = graph_backend.overlaps()
    assert len(overlaps) == 1
    assert [node1, node2] in overlaps


def test_compute_overlaps_with_threshold(graph_backend: BaseGraph) -> None:
    """Test compute_overlaps with different IoU thresholds."""
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Create masks with different overlap levels
    mask1_data = np.array([[True, True], [True, True]], dtype=bool)
    mask1 = Mask(mask1_data, bbox=np.array([0, 0, 2, 2]))

    # Partially overlapping mask (IoU = 0.5)
    mask2_data = np.array([[True, True], [False, False]], dtype=bool)
    mask2 = Mask(mask2_data, bbox=np.array([0, 0, 2, 2]))

    # Non-overlapping mask
    mask3_data = np.array([[True, True], [True, True]], dtype=bool)
    mask3 = Mask(mask3_data, bbox=np.array([10, 10, 12, 12]))

    node1 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask1})
    node2 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask2})
    graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask3})

    # With threshold 0.7, no overlaps should be found (IoU = 0.5 < 0.7)
    graph_backend.compute_overlaps(iou_threshold=0.7)
    overlaps = graph_backend.overlaps()
    valid_overlaps = [o for o in overlaps if None not in o]
    assert len(valid_overlaps) == 0

    # With threshold 0.3, mask1 and mask2 should overlap
    graph_backend.compute_overlaps(iou_threshold=0.3)
    overlaps = graph_backend.overlaps()
    valid_overlaps = [o for o in overlaps if None not in o]
    assert len(valid_overlaps) == 1
    assert [node1, node2] in valid_overlaps


def test_compute_overlaps_multiple_timepoints(graph_backend: BaseGraph) -> None:
    """Test compute_overlaps across multiple time points."""
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)

    # Time 0: overlapping masks
    mask1_t0 = Mask(np.array([[True, True], [True, True]], dtype=bool), bbox=np.array([0, 0, 2, 2]))
    mask2_t0 = Mask(np.array([[True, True], [False, False]], dtype=bool), bbox=np.array([0, 0, 2, 2]))

    # Time 1: non-overlapping masks
    mask1_t1 = Mask(np.array([[True, True], [True, True]], dtype=bool), bbox=np.array([0, 0, 2, 2]))
    mask2_t1 = Mask(np.array([[True, True], [True, True]], dtype=bool), bbox=np.array([10, 10, 12, 12]))

    node1_t0 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask1_t0})
    node2_t0 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.MASK: mask2_t0})
    graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.MASK: mask1_t1})
    graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.MASK: mask2_t1})

    graph_backend.compute_overlaps(iou_threshold=0.3)

    overlaps = graph_backend.overlaps()
    valid_overlaps = [o for o in overlaps if None not in o]
    assert len(valid_overlaps) == 1
    assert [node1_t0, node2_t0] in valid_overlaps


def test_compute_overlaps_invalid_threshold(graph_backend: BaseGraph) -> None:
    """Test compute_overlaps with invalid threshold values."""
    with pytest.raises(ValueError, match="iou_threshold must be between 0.0 and 1.0"):
        graph_backend.compute_overlaps(iou_threshold=-0.1)

    with pytest.raises(ValueError, match="iou_threshold must be between 0.0 and 1.0"):
        graph_backend.compute_overlaps(iou_threshold=1.1)


def test_compute_overlaps_empty_graph(graph_backend: BaseGraph) -> None:
    """Test compute_overlaps on empty graph."""
    graph_backend.compute_overlaps(iou_threshold=0.5)
    assert not graph_backend.has_overlaps()
    assert graph_backend.overlaps() == []


def test_summary(graph_backend: BaseGraph) -> None:
    """Test summary method."""
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)
    graph_backend.add_edge_attr_key("type", "good")

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 0, "x": 3.0})

    graph_backend.add_edge(node1, node2, {"weight": 0.5, "type": "good"})
    graph_backend.add_edge(node3, node2, {"weight": 0.5, "type": "bad"})

    graph_backend.add_overlap(node1, node3)

    summary = graph_backend.summary(attrs_stats=True, print_summary=True)
    print(summary)

    assert isinstance(summary, str)
    assert "Graph summary" in summary
    assert "Number of nodes" in summary
    assert "Number of edges" in summary


def test_spatial_filter_basic(graph_backend: BaseGraph) -> None:
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_node_attr_key("z", 0.0)
    graph_backend.add_node_attr_key("bbox", None)

    node1 = graph_backend.add_node({"t": 0, "x": 1.0, "y": 1.0, "z": 1.0, "bbox": np.array([6, 6, 6, 8, 8, 8])})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0, "z": 2.0, "bbox": np.array([0, 0, 0, 3, 3, 3])})
    node3 = graph_backend.add_node({"t": 0, "x": 3.0, "y": 3.0, "z": 3.0, "bbox": np.array([2, 2, 2, 4, 4, 4])})

    graph_backend.add_edge(node1, node2, attrs={})
    graph_backend.add_edge(node3, node2, attrs={})

    g_filter = graph_backend.spatial_filter(["t", "x", "y", "z"])
    bb_filter = graph_backend.bbox_spatial_filter("t", "bbox")

    assert set(g_filter[0:0, 0:2, 0:2, 0:2].node_ids()) == {node1}
    assert set(bb_filter[0:0, 0:2, 0:2, 0:2].node_ids()) == {node3}


def test_assign_track_ids(graph_backend: BaseGraph):
    if isinstance(graph_backend, SQLGraph):
        pytest.skip("`assign_track_ids` is not available for `SQLGraph`")
    else:
        # Add nodes:
        #     0
        #    / \
        #   1   2
        nodes = [
            graph_backend.add_node({DEFAULT_ATTR_KEYS.T: 0}),
            graph_backend.add_node({DEFAULT_ATTR_KEYS.T: 1}),
            graph_backend.add_node({DEFAULT_ATTR_KEYS.T: 1}),
        ]
        graph_backend.add_edge(nodes[0], nodes[1], {})
        graph_backend.add_edge(nodes[0], nodes[2], {})

    tracks_graph = graph_backend.assign_track_ids()
    track_ids = graph_backend.node_attrs(attr_keys=[DEFAULT_ATTR_KEYS.TRACK_ID])
    assert len(track_ids) == 3
    assert len(set(track_ids[DEFAULT_ATTR_KEYS.TRACK_ID])) == 3
    # assert len

    assert isinstance(tracks_graph, rx.PyDiGraph)
    assert tracks_graph.num_nodes() == 3 + 1  # Three tracks (includes null node (0))


def test_tracklet_graph_basic(graph_backend: BaseGraph) -> None:
    """Test basic tracklet_graph functionality."""
    # Add track_id attribute and nodes with track IDs
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACK_ID, -1)

    # Create nodes with different track IDs
    node1 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.TRACK_ID: 1})
    node2 = graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.TRACK_ID: 2})
    node3 = graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.TRACK_ID: 3})
    node4 = graph_backend.add_node({"t": 2, DEFAULT_ATTR_KEYS.TRACK_ID: 2})
    node5 = graph_backend.add_node({"t": 2, DEFAULT_ATTR_KEYS.TRACK_ID: 3})
    node6 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.TRACK_ID: 4})
    node7 = graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.TRACK_ID: 4})
    node8 = graph_backend.add_node({"t": 2, DEFAULT_ATTR_KEYS.TRACK_ID: 4})

    graph_backend.add_edge_attr_key("weight", 0.0)

    # Add edges within tracks (will be filtered out)
    graph_backend.add_edge(node1, node2, {"weight": 0.8})
    graph_backend.add_edge(node1, node3, {"weight": 0.9})
    graph_backend.add_edge(node2, node4, {"weight": 0.5})
    graph_backend.add_edge(node3, node5, {"weight": 0.7})

    graph_backend.add_edge(node6, node7, {"weight": 0.9})
    graph_backend.add_edge(node7, node8, {"weight": 0.5})

    # Create tracklet graph
    tracklet_graph = graph_backend.tracklet_graph()

    # The method adds all track IDs from nodes_df, including duplicates
    # So we should have 4 nodes (one for each node in original graph)
    assert tracklet_graph.num_nodes() == 4
    assert tracklet_graph.num_edges() == 2

    # node content is the same as node ids
    nodes = tracklet_graph.nodes()
    assert set(nodes) == {1, 2, 3, 4}

    edges = tracklet_graph.edges()
    assert set(edges) == {(1, 2), (1, 3)}


def test_tracklet_graph_with_ignore_track_id(graph_backend: BaseGraph) -> None:
    """Test tracklet_graph with ignore_track_id parameter."""
    # Add track_id attribute and nodes with track IDs
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACK_ID, -1)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Simple test case: just check that the method accepts the parameter
    # and filters out nodes properly when there are no edges
    node1 = graph_backend.add_node({"t": 0, DEFAULT_ATTR_KEYS.TRACK_ID: 1})
    node2 = graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.TRACK_ID: 2})
    node3 = graph_backend.add_node({"t": 1, DEFAULT_ATTR_KEYS.TRACK_ID: -1})

    graph_backend.add_edge(node1, node2, {"weight": 0.8})
    graph_backend.add_edge(node1, node3, {"weight": 0.9})

    # Test that tracklet_graph method accepts ignore_track_id parameter
    tracklet_graph = graph_backend.tracklet_graph(ignore_track_id=-1)
    assert tracklet_graph.num_nodes() == 2
    assert tracklet_graph.num_edges() == 1

    assert set(tracklet_graph.nodes()) == {1, 2}
    assert set(tracklet_graph.edges()) == {(1, 2)}


def test_tracklet_graph_missing_track_id_key(graph_backend: BaseGraph) -> None:
    """Test tracklet_graph raises error when track_id_key doesn't exist."""
    with pytest.raises(ValueError, match="Track id key 'track_id' not found in graph"):
        graph_backend.tracklet_graph()


def test_nodes_interface(graph_backend: BaseGraph) -> None:
    graph_backend.add_node_attr_key("x", 0)

    # Simple test case: just check that the method accepts the parameter
    # and filters out nodes properly when there are no edges
    node1 = graph_backend.add_node({"t": 0, "x": 1})
    node2 = graph_backend.add_node({"t": 1, "x": 0})
    node3 = graph_backend.add_node({"t": 2, "x": -1})

    assert graph_backend[node1]["x"] == 1
    assert graph_backend[node2]["x"] == 0
    assert graph_backend[node3]["x"] == -1

    graph_backend.add_node_attr_key("y", -1)

    graph_backend[node2]["y"] = 5

    assert graph_backend[node1]["y"] == -1
    assert graph_backend[node2]["y"] == 5
    assert graph_backend[node3]["y"] == -1

    assert graph_backend[node1].to_dict() == {"t": 0, "x": 1, "y": -1}
    assert graph_backend[node2].to_dict() == {"t": 1, "x": 0, "y": 5}
    assert graph_backend[node3].to_dict() == {"t": 2, "x": -1, "y": -1}


def test_custom_indices(graph_backend: BaseGraph) -> None:
    """Test custom node indices functionality."""

    if not graph_backend.supports_custom_indices:
        pytest.skip("Graph does not support custom indices")

    # Add attribute keys for testing
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)

    # Test add_node with custom index
    custom_node_id = graph_backend.add_node({"t": 0, "x": 10.0, "y": 20.0}, index=12345)
    assert custom_node_id == 12345

    # Test add_node without custom index (auto-generated)
    auto_node_id = graph_backend.add_node({"t": 0, "x": 15.0, "y": 25.0})
    assert auto_node_id != 12345  # Should be different from custom

    # Test bulk_add_nodes with custom indices
    nodes = [{"t": 1, "x": 30.0, "y": 40.0}, {"t": 1, "x": 35.0, "y": 45.0}, {"t": 1, "x": 40.0, "y": 50.0}]
    custom_indices = [50000, 60000, 70000]

    returned_indices = graph_backend.bulk_add_nodes(nodes, indices=custom_indices)
    assert returned_indices == custom_indices

    # Test bulk_add_nodes without custom indices (auto-generated)
    auto_nodes = [{"t": 2, "x": 100.0, "y": 200.0}, {"t": 2, "x": 150.0, "y": 250.0}]
    auto_indices = graph_backend.bulk_add_nodes(auto_nodes)
    assert len(auto_indices) == 2
    assert all(idx not in custom_indices for idx in auto_indices)

    # Verify all nodes exist in the graph
    all_node_ids = graph_backend.node_ids()
    expected_ids = [custom_node_id, auto_node_id, *custom_indices, *auto_indices]
    for expected_id in expected_ids:
        assert expected_id in all_node_ids, f"Node ID {expected_id} not found in graph"

    # Test that custom indices work with queries
    custom_node_df = graph_backend.filter(node_ids=[12345]).node_attrs(attr_keys=["x", "y"])
    assert len(custom_node_df) == 1
    assert custom_node_df["x"].to_list()[0] == 10.0
    assert custom_node_df["y"].to_list()[0] == 20.0

    # Test bulk_add_nodes with mismatched indices length
    with pytest.raises(ValueError, match="Length of indices .* must match length of nodes"):
        graph_backend.bulk_add_nodes([{"t": 3, "x": 1.0, "y": 1.0}], indices=[1, 2, 3])


def test_remove_node(graph_backend: BaseGraph) -> None:
    """Test removing nodes from the graph."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Add nodes
    node1 = graph_backend.add_node({"t": 0, "x": 1.0, "y": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0, "y": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0, "y": 3.0})

    # Add edges
    edge1 = graph_backend.add_edge(node1, node2, {"weight": 0.5})
    edge2 = graph_backend.add_edge(node2, node3, {"weight": 0.7})
    edge3 = graph_backend.add_edge(node1, node3, {"weight": 0.3})

    # Add overlap
    graph_backend.add_overlap(node1, node3)

    initial_node_count = graph_backend.num_nodes

    # Remove node2 - this should also remove edges involving node2
    graph_backend.remove_node(node2)

    # Check node count decreased
    assert graph_backend.num_nodes == initial_node_count - 1

    # Check that node2 is no longer in the graph
    assert node2 not in graph_backend.node_ids()
    assert node1 in graph_backend.node_ids()
    assert node3 in graph_backend.node_ids()

    # Check that edges involving node2 were removed
    remaining_edges = graph_backend.edge_attrs()
    remaining_edge_ids = set(remaining_edges[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())

    # Only edge3 (node1->node3) should remain
    assert edge1 not in remaining_edge_ids
    assert edge2 not in remaining_edge_ids
    assert edge3 in remaining_edge_ids

    # Check that the remaining edge is correct
    assert len(remaining_edges) == 1
    assert remaining_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[0] == node1
    assert remaining_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[0] == node3
    assert remaining_edges["weight"].to_list()[0] == 0.3

    # Check that overlaps involving node2 are removed, but others remain
    remaining_overlaps = graph_backend.overlaps()
    # The overlap (node1, node3) should still exist since node2 wasn't involved
    assert [node1, node3] in remaining_overlaps

    # Test error when removing non-existent node
    with pytest.raises(ValueError, match="Node .* does not exist in the graph"):
        graph_backend.remove_node(99999)

    # Test error when removing already removed node
    with pytest.raises(ValueError, match="Node .* does not exist in the graph"):
        graph_backend.remove_node(node2)


def test_remove_node_and_add_new_nodes(graph_backend: BaseGraph) -> None:
    """Test removing nodes and then adding new nodes."""
    # Add attribute keys
    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_edge_attr_key("weight", 0.0)

    # Add initial nodes
    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0})

    # Add edges
    graph_backend.add_edge(node1, node2, {"weight": 0.5})
    graph_backend.add_edge(node2, node3, {"weight": 0.7})

    initial_node_count = graph_backend.num_nodes
    initial_edge_count = graph_backend.num_edges

    # Remove the middle node
    graph_backend.remove_node(node2)

    assert graph_backend.num_nodes == initial_node_count - 1
    assert graph_backend.num_edges == initial_edge_count - 2  # Both edges removed

    # Add new nodes after removal
    new_node1 = graph_backend.add_node({"t": 1, "x": 10.0})
    new_node2 = graph_backend.add_node({"t": 3, "x": 20.0})

    assert graph_backend.num_nodes == initial_node_count + 1  # net +1 node

    # Add new edges with new nodes
    graph_backend.add_edge(node1, new_node1, {"weight": 0.9})
    graph_backend.add_edge(new_node1, node3, {"weight": 0.8})
    graph_backend.add_edge(node3, new_node2, {"weight": 0.6})

    assert graph_backend.num_edges == initial_edge_count + 1  # net +1 edge

    # Verify the graph structure by checking node attributes
    # Since RustWorkX can reuse node IDs, we need to check attributes instead of just IDs
    node_attrs = graph_backend.node_attrs()

    # Create sets of (t, x) tuples for comparison
    current_nodes = set(zip(node_attrs["t"].to_list(), node_attrs["x"].to_list(), strict=True))

    # Check that the original nodes (except node2) are present
    assert (0, 1.0) in current_nodes  # node1 attributes
    assert (2, 3.0) in current_nodes  # node3 attributes
    assert (1, 2.0) not in current_nodes  # node2 attributes should be gone

    # Check that new nodes are present
    assert (1, 10.0) in current_nodes  # new_node1 attributes
    assert (3, 20.0) in current_nodes  # new_node2 attributes

    # Verify edge count is correct
    assert graph_backend.num_edges == initial_edge_count + 1  # net +1 edge

    # Test time points are updated correctly
    time_points = set(graph_backend.time_points())
    assert time_points == {0, 1, 2, 3}  # all time points represented


def test_remove_isolated_node(graph_backend: BaseGraph) -> None:
    """Test removing a node with no edges."""
    # Add an isolated node
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})  # isolated node

    initial_count = graph_backend.num_nodes

    # Remove the isolated node
    graph_backend.remove_node(node2)

    assert graph_backend.num_nodes == initial_count - 1
    assert node1 in graph_backend.node_ids()
    assert node2 not in graph_backend.node_ids()


def test_remove_all_nodes_in_time_point(graph_backend: BaseGraph) -> None:
    """Test that time points are cleaned up when all nodes in a time are removed."""
    # Add nodes at different time points
    graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})
    node3 = graph_backend.add_node({"t": 1})  # another node at t=1
    graph_backend.add_node({"t": 2})

    initial_time_points = set(graph_backend.time_points())
    assert initial_time_points == {0, 1, 2}

    # Remove one node from t=1
    graph_backend.remove_node(node2)
    time_points_after_one = set(graph_backend.time_points())
    assert time_points_after_one == {0, 1, 2}  # t=1 still has node3

    # Remove the other node from t=1
    graph_backend.remove_node(node3)
    time_points_after_two = set(graph_backend.time_points())
    assert time_points_after_two == {0, 2}  # t=1 should be gone


def test_geff_roundtrip(graph_backend: BaseGraph) -> None:
    """Test geff roundtrip."""

    graph_backend.add_node_attr_key("x", 0.0)
    graph_backend.add_node_attr_key("y", 0.0)
    graph_backend.add_node_attr_key("z", 0.0)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.BBOX, None)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.MASK, None)
    graph_backend.add_node_attr_key(DEFAULT_ATTR_KEYS.TRACK_ID, -1)

    graph_backend.add_edge_attr_key("weight", 0.0)

    node1 = graph_backend.add_node(
        {
            "t": 0,
            "x": 1.0,
            "y": 1.0,
            "z": 1.0,
            "bbox": np.array([6, 6, 8, 8]),
            "mask": Mask(np.array([[True, True], [True, True]], dtype=bool), bbox=np.array([6, 6, 8, 8])),
            DEFAULT_ATTR_KEYS.TRACK_ID: 1,
        }
    )
    node2 = graph_backend.add_node(
        {
            "t": 1,
            "x": 2.0,
            "y": 2.0,
            "z": 2.0,
            "bbox": np.array([0, 0, 3, 3]),
            "mask": Mask(
                np.array([[True, True, True], [True, True, True], [True, True, True]], dtype=bool),
                bbox=np.array([0, 0, 3, 3]),
            ),
            DEFAULT_ATTR_KEYS.TRACK_ID: 1,
        }
    )

    node3 = graph_backend.add_node(
        {
            "t": 2,
            "x": 3.0,
            "y": 3.0,
            "z": 3.0,
            "bbox": np.array([2, 2, 4, 4]),
            "mask": Mask(np.array([[True, True], [True, True]], dtype=bool), bbox=np.array([2, 2, 4, 4])),
            DEFAULT_ATTR_KEYS.TRACK_ID: 1,
        }
    )

    graph_backend.add_edge(node1, node2, {"weight": 0.8})
    graph_backend.add_edge(node2, node3, {"weight": 0.9})

    output_store = MemoryStore()

    graph_backend.to_geff(geff_store=output_store)

    geff_graph = RustWorkXGraph.from_geff(output_store)

    assert geff_graph.num_nodes == 3
    assert geff_graph.num_edges == 2

    rx_graph = graph_backend.filter().subgraph().rx_graph

    assert set(graph_backend.node_attr_keys) == set(geff_graph.node_attr_keys)
    assert set(graph_backend.edge_attr_keys) == set(geff_graph.edge_attr_keys)

    assert rx.is_isomorphic(
        rx_graph,
        geff_graph.rx_graph,
    )
